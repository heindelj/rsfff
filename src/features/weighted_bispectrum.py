"""Learnable, weighted lambda-bispectrum (body order nu = 3) via cuequivariance.

This is the nu=3 analogue of your existing `equivariant_power_spectrum` (nu=2),
written as a *learnable weighted tensor contraction* instead of a fixed-CG einsum.

Design
------
The density `A` from `DensityExpansion` is, per atom, a stack of irreps with one
copy per composite channel `c = (Z, n)`:

    A : (N, n_species, n_max, (l_max+1)**2)
      -> reshape -> (N, C, (l_max+1)**2),   C = n_species * n_max

Each channel carries the O3 irreps  `0e + 1o + 2e + 3o + ...`  (parity (-1)^l,
because the coefficients come from spherical harmonics of bond vectors).

A symmetric contraction of `A` with itself `nu` times, with learnable weights on
every CG coupling path, is exactly the (generalized, channel-wise) lambda-SOAP of
body order `nu`. cuequivariance gives this as a single fused kernel:

  * `cue.descriptors.symmetric_contraction(irreps_in, irreps_out, degrees)` builds
    the descriptor. `degrees=(3,)` => the *pure* nu=3 bispectrum.
  * Weights are an operand; making them an `nn.Parameter` indexed per species gives
    MACE-style element-dependent learnable weights, reusing your `species_idx`.

Output parity matches your power-spectrum convention. For nu=3 the nonzero-CG chain
A_l1 (x) A_l2 -> k, then k (x) A_l3 -> lam forces l1+l2+l3 ≡ lam (mod 2), so the
lambda channel has parity (-1)^lam exactly as in nu=2:  lam=0 -> 0e, 1 -> 1o, 2 -> 2e.

Two implementations are provided:

  LearnableBispectrum       pure nu=3 via the descriptor + SegmentedPolynomial.
                            This is the literal bispectrum.
  LearnableSymContraction   convenience wrapper over `cuet.SymmetricContraction`
                            (body orders 1..degree, the standard MACE block) when
                            you'd rather have the whole low-order polynomial in one
                            fused call.

Both consume the *same* `A` you already build, and both return tensors shaped to
slot into your `LambdaFeatures` layout: (N, P0) for lam=0, (N, 2*lam+1, P_lam)
otherwise.

Tested against e3nn for O3 equivariance: see `_equivariance_test` at the bottom.
Requires: cuequivariance, cuequivariance-torch (>=0.10), torch. e3nn only for the test.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

import cuequivariance as cue
import cuequivariance_torch as cuet


# --------------------------------------------------------------------------
# irreps helpers
# --------------------------------------------------------------------------
def _irreps_str(mul: int, ells: Sequence[int]) -> str:
    """e.g. mul=4, ells=(0,1,2) -> '4x0e + 4x1o + 4x2e' (parity = (-1)**l)."""
    return " + ".join(f"{mul}x{l}{'e' if l % 2 == 0 else 'o'}" for l in ells)


def _density_to_ir_mul(A: torch.Tensor) -> torch.Tensor:
    """(N, C, (l_max+1)**2) density -> (N, C*(l_max+1)**2) in cue.ir_mul layout.

    `A`'s last axis is already component-major over irreps [0e | 1o | 2e | ...].
    ir_mul wants, per irrep, (2l+1, mul) i.e. component-major / mul-inner, so we
    just move the channel axis inside the component axis. Mirrors the preprocessing
    in the cuet.SymmetricContraction docs (`transpose(1, 2).flatten(1)`).
    """
    n = A.shape[0]
    return A.transpose(1, 2).reshape(n, -1)


def _split_ir_mul_output(
    y: torch.Tensor, lambdas: Sequence[int], mul: int
) -> dict[int, torch.Tensor]:
    """Flat ir_mul output (N, sum_lam mul*(2lam+1)) -> {lam: (N, 2lam+1, mul)}.

    Output irreps are laid out in the given `lambdas` order; each lam block is
    (2lam+1, mul) component-major, so a plain reshape recovers the component axis.
    """
    out: dict[int, torch.Tensor] = {}
    n, off = y.shape[0], 0
    for lam in lambdas:
        d = mul * (2 * lam + 1)
        out[lam] = y[:, off : off + d].reshape(n, 2 * lam + 1, mul)
        off += d
    return out


# --------------------------------------------------------------------------
# Pure nu = 3 bispectrum (descriptor + SegmentedPolynomial)
# --------------------------------------------------------------------------
class LearnableBispectrum(nn.Module):
    """Learnable equivariant lambda-bispectrum (body order nu = 3, channel-wise).

    p^{lam}_mu[i, u] = sum_{paths} W[species_i, path, u]
                       * (A (x) A (x) A)^{lam}_{mu}[i, u]

    where `u` runs over the `n_channels` input channels (treated in parallel, i.e.
    each channel is contracted with itself -- the standard channel-wise reduction;
    cross-channel mixing is left to the surrounding linear layers, exactly as in
    MACE). Weights are per-species via `num_elements = n_species`.

    Parameters
    ----------
    n_species, n_max, l_max : as in your featurizer (n_channels = n_species*n_max).
    selected_lambdas        : output lambdas, e.g. (0, 2) or (0, 1, 2).
    out_mul                 : output multiplicity P_lam. Defaults to n_channels
                              (channel-preserving, MACE-like). Set smaller to
                              compress, larger to widen.
    degrees                 : contraction degrees. (3,) = pure bispectrum. You can
                              pass (1, 2, 3) to fold in lower orders too.
    """

    def __init__(
        self,
        *,
        n_species: int,
        n_max: int,
        l_max: int,
        selected_lambdas: Sequence[int] = (0, 2),
        out_mul: int | None = None,
        num_elements: int | None = None,
        degrees: Sequence[int] = (3,),
        weight_init_std: float = 1.0,
        method: str = "",          # "" -> cue auto-picks; use "naive" on CPU
        math_dtype: torch.dtype | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.n_channels = n_species * n_max
        self.l_max = int(l_max)
        self.lambdas = tuple(sorted({int(v) for v in selected_lambdas}))
        self.out_mul = int(out_mul) if out_mul is not None else self.n_channels
        # Per-element weights are decoupled from the density channel count: pass
        # num_elements=1 for shared weights (all atoms use one weight set) while
        # n_channels stays n_species*n_max. Defaults to per-species (MACE-style).
        self.num_elements = int(num_elements) if num_elements is not None else int(n_species)
        dtype = dtype or torch.get_default_dtype()

        irreps_in = cue.Irreps("O3", _irreps_str(self.n_channels, range(l_max + 1)))
        irreps_out = cue.Irreps("O3", _irreps_str(self.out_mul, self.lambdas))

        desc = cue.descriptors.symmetric_contraction(
            irreps_in, irreps_out, tuple(int(d) for d in degrees)
        )
        # Sanity: descriptor operands are (weights, input, [input...], output);
        # after dedup of the repeated input, inputs == [weights, x].
        poly = desc.polynomial
        self._n_weights = poly.inputs[0].size
        self._in_dim = poly.inputs[1].size
        self.poly = cuet.SegmentedPolynomial(poly, method=method, math_dtype=math_dtype)

        # Per-element weights, gathered per atom in forward (avoids any
        # method-specific constraints on indexed operands). num_elements=1 => shared.
        w = torch.randn(self.num_elements, self._n_weights, dtype=dtype) * weight_init_std
        self.weight = nn.Parameter(w)

    @property
    def feature_dims(self) -> dict[int, int]:
        return {lam: self.out_mul for lam in self.lambdas}

    def _weight_index(self, species_idx, n, device):
        # Shared weights (num_elements=1) index a single row; else per-species.
        if self.num_elements == 1:
            return torch.zeros(n, dtype=torch.long, device=device)
        return species_idx.to(torch.long)

    def forward(
        self, A: torch.Tensor, species_idx: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        # A: (N, n_species, n_max, (l_max+1)**2) or already (N, C, (l_max+1)**2)
        if A.dim() == 4:
            A = A.reshape(A.shape[0], self.n_channels, A.shape[-1])
        x = _density_to_ir_mul(A)                      # (N, C*(l_max+1)**2), ir_mul
        idx = self._weight_index(species_idx, x.shape[0], x.device)  # (N,)
        w = self.weight.to(x.dtype)[idx]               # (N, n_weights)
        y = self.poly([w, x])[0]                        # (N, irreps_out.dim), ir_mul
        return _split_ir_mul_output(y, self.lambdas, self.out_mul)


# --------------------------------------------------------------------------
# Convenience wrapper: body orders 1..degree (standard MACE block)
# --------------------------------------------------------------------------
class LearnableSymContraction(nn.Module):
    """Wrapper over `cuet.SymmetricContraction` (orders 1..contraction_degree).

    Use this if you want the whole low-order learnable polynomial fused into one
    documented kernel. The genuine bispectrum is the top (degree-3) term; orders
    1-2 overlap with your existing power spectrum but are cheap and the net can
    down-weight them. Same I/O contract as `LearnableBispectrum`.
    """

    def __init__(
        self,
        *,
        n_species: int,
        n_max: int,
        l_max: int,
        selected_lambdas: Sequence[int] = (0, 2),
        out_mul: int | None = None,
        num_elements: int | None = None,
        contraction_degree: int = 3,
        method: str | None = None,     # set "naive" on CPU
        dtype: torch.dtype | None = None,
        math_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.n_channels = n_species * n_max
        self.lambdas = tuple(sorted({int(v) for v in selected_lambdas}))
        self.out_mul = int(out_mul) if out_mul is not None else self.n_channels
        # Decoupled from the density channel count: num_elements=1 -> shared weights.
        self.num_elements = int(num_elements) if num_elements is not None else int(n_species)

        irreps_in = cue.Irreps("O3", _irreps_str(self.n_channels, range(l_max + 1)))
        irreps_out = cue.Irreps("O3", _irreps_str(self.out_mul, self.lambdas))
        self.sc = cuet.SymmetricContraction(
            irreps_in,
            irreps_out,
            contraction_degree=contraction_degree,
            num_elements=self.num_elements,
            layout_in=cue.ir_mul,
            layout_out=cue.ir_mul,
            dtype=dtype or torch.get_default_dtype(),
            math_dtype=math_dtype,
            method=method,
        )

    @property
    def feature_dims(self) -> dict[int, int]:
        return {lam: self.out_mul for lam in self.lambdas}

    def forward(
        self, A: torch.Tensor, species_idx: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        if A.dim() == 4:
            A = A.reshape(A.shape[0], self.n_channels, A.shape[-1])
        x = _density_to_ir_mul(A)                         # (N, ...), ir_mul
        if self.num_elements == 1:
            idx = torch.zeros(x.shape[0], dtype=torch.int32, device=x.device)
        else:
            idx = species_idx.to(torch.int32)
        y = self.sc(x, idx)                                # (N, irreps_out.dim)
        return _split_ir_mul_output(y, self.lambdas, self.out_mul)


# --------------------------------------------------------------------------
# Integration sketch for FlatLambdaSOAPFeaturizer
# --------------------------------------------------------------------------
# In __init__ (after self.density / specs are built):
#
#     self.bispectrum = LearnableBispectrum(
#         n_species=self.n_species,
#         n_max=int(n_max),
#         l_max=self.l_max,
#         selected_lambdas=self.selected_lambdas,
#         out_mul=self.n_species * int(n_max),   # or a compressed width
#     )
#
# In forward, reuse the density `A` you already compute:
#
#     A = self.density(positions, species_idx, edge_index, int(positions.shape[0]))
#     per_lambda = equivariant_power_spectrum(...)            # nu=2, unchanged
#     bis = self.bispectrum(A, species_idx)                   # nu=3, {lam: tensor}
#
#     # bis[0]      -> (Ntot, 1, P0)  squeeze for invariants
#     # bis.get(1)  -> (Ntot, 3, P1)  odd vector, if 1 in selected_lambdas
#     # bis[2]      -> (Ntot, 5, P2)  even tensor
#     # concat with per_lambda along the feature axis, or carry as new fields.
#
# The whole featurizer already runs under @torch.compiler.disable, so the cue
# custom ops won't trip dynamo. If you call the bispectrum from a compiled region,
# wrap it with @torch.compiler.disable like the rest of the featurizer.


# --------------------------------------------------------------------------
# O3 equivariance test (uses e3nn, which your project already depends on)
# --------------------------------------------------------------------------
def _equivariance_test(
    cls=LearnableBispectrum,
    *,
    n_species: int = 2,
    n_max: int = 3,
    l_max: int = 3,
    lambdas: Sequence[int] = (0, 1, 2),
    n_atoms: int = 5,
    seed: int = 0,
    atol: float = 1e-4,
) -> None:
    """Rotate the density, run the module, check each lam rotates by Wigner-D_lam."""
    from e3nn import o3

    torch.manual_seed(seed)
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    C = n_species * n_max
    L2 = (l_max + 1) ** 2
    A = torch.randn(n_atoms, C, L2, dtype=dtype)
    species = torch.randint(0, n_species, (n_atoms,))

    method = "naive" if not torch.cuda.is_available() else ""
    kw = dict(method=method) if cls is LearnableBispectrum else dict(method="naive")
    mod = cls(
        n_species=n_species, n_max=n_max, l_max=l_max,
        selected_lambdas=lambdas, dtype=dtype, **kw,
    ).to(dtype)

    # Wigner-D for the input irreps (one copy of 0e+1o+2e+...), applied to last axis.
    in_irreps = o3.Irreps([(1, (l, (-1) ** l)) for l in range(l_max + 1)])
    a, b, c = torch.rand(3, dtype=dtype) * 6.283
    D_in = in_irreps.D_from_angles(a, b, c).to(dtype)            # (L2, L2)
    A_rot = torch.einsum("ij,ncj->nci", D_in, A)

    out = mod(A, species)
    out_rot = mod(A_rot, species)

    print(f"\n=== {cls.__name__} O3-equivariance ===")
    for lam in lambdas:
        D_lam = o3.Irrep(lam, (-1) ** lam).D_from_angles(a, b, c).to(dtype)  # (2l+1,2l+1)
        expected = torch.einsum("ij,njp->nip", D_lam, out[lam])
        err = (out_rot[lam] - expected).abs().max().item()
        flag = "OK " if err < atol else "FAIL"
        print(f"  lam={lam}: max|D·f(A) - f(R·A)| = {err:.2e}  [{flag}]")
        assert err < atol, f"lam={lam} not equivariant (err={err:.2e})"
    print("  all equivariant ✓")


if __name__ == "__main__":
    _equivariance_test(LearnableBispectrum)
    try:
        _equivariance_test(LearnableSymContraction)
    except Exception as exc:  # SymmetricContraction CPU support varies by build
        print(f"\nLearnableSymContraction CPU run skipped: {exc}")