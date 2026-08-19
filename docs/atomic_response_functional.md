# Including Atomic Energies in the SCF Functional

## 1. Electronic and Geometric Atomic Features

Each atom begins with a geometry-dependent equivariant representation, for example from a MACE-style network,

[
h_i^{\mathrm{geom}} = \mathrm{MACE}(Z,R)_i.
]

This representation describes the local geometric and chemical environment but does not yet contain the self-consistent electronic state.

The electronic degrees of freedom are the atomic multipoles,

[
M_i = \left(q_i,\boldsymbol{\mu}_i,\mathbf Q_i,\ldots\right),
]

together with their conjugate electrostatic quantities,

[
\Phi_i =
\left(
V_i,\mathbf E_i,\nabla\mathbf E_i,\ldots
\right).
]

Here (V_i) is the electrostatic potential, (\mathbf E_i) is the electric field, and (\nabla\mathbf E_i) is the field gradient at atom (i).

There is no need to make a fundamental distinction between "permanent" and "induced" multipoles in the representation passed to the neural network. Those labels describe how a model constructs the multipoles, rather than distinct physical observables. The relevant electronic state is simply the final set of self-consistent multipoles (M_i).

---

## 2. Atomic Energy as Part of the SCF Functional

The important coupling is obtained by making the atomic energy explicitly dependent on the electronic variables being optimized.

Instead of first solving a polarization model and subsequently evaluating an unrelated atomic energy,

[
M^* \rightarrow E_{\mathrm{atom}}(M^*),
]

the atomic energy is included directly in the functional whose stationary point defines (M^*):

[
E_{\mathrm{SCF}}(R,M)
=====================

E_{\mathrm{FF}}(R,M)
+
\sum_i E_i^{\mathrm{atom}}(h_i,M_i,\Phi_i).
]

The self-consistent multipoles are therefore determined by

[
M^*
===

\arg\min_M E_{\mathrm{SCF}}(R,M),
]

subject to the appropriate charge-conservation constraints.

Consequently, the condition

[
\frac{\partial E_{\mathrm{SCF}}}{\partial M}=0
]

contains contributions from both the classical multipolar interactions and the learned atomic energy.

This creates the desired two-way coupling:

[
\text{multipoles}
\longrightarrow
\text{atomic energies}
]

and simultaneously

[
\text{atomic energies}
\longrightarrow
\text{optimal multipoles}.
]

The SCF electronic state therefore minimizes the **total learned energy**, rather than a polarization functional that is only indirectly related to the bonding energy.

---

## 3. Charge Flux and Charge Conservation

For the monopole sector, it is useful to parameterize redistribution using charge fluxes rather than unconstrained atomic charges.

For example,

[
\mathbf q
=========

\mathbf q^{(0)} + B\mathbf p,
]

where (\mathbf p) contains charge-transfer amplitudes and (B) is an incidence matrix describing the allowed transfer channels.

Because

[
\mathbf 1^T B=0,
]

the total charge is conserved automatically:

[
\sum_i q_i = Q_{\mathrm{tot}}.
]

The charge-flux variables can be included directly among the SCF degrees of freedom. The functional therefore becomes schematically

[
E_{\mathrm{SCF}}
================

E_{\mathrm{multipole}}(M)
+
E_{\mathrm{flux}}(p)
+
\sum_i E_i^{\mathrm{atom}}(h_i,M_i,\Phi_i),
]

with

[
q=q(p).
]

The learned atomic-energy contribution then influences how charge is redistributed, while charge redistribution modifies the atomic energies.

---

## 5. Constructing Fragment Features

The fragment representation should be constructed from geometry-aware atomic features rather than directly from element embeddings.

For fragment (F),

[
f_F
===

\mathrm{FragmentEncoder}
\left(
{h_i^{\mathrm{geom}}:i\in F},
s_F
\right),
]

where (s_F) contains the known fragment electronic state, such as charge and spin.

A useful construction combines invariant pooling with attention:

[
u_F
===

\sum_{i\in F}\phi(h_{i,0}),
]

[
v_F
===

\mathrm{Attention}
\left(
s_F,
{h_{i,0}},
{h_{i,0}}
\right),
]

followed by

[
f_F
===

\rho(s_F,u_F,v_F).
]

Here (h_{i,0}) denotes the scalar part of the equivariant atomic representation.

The summed representation preserves extensive information such as composition and atom count, while attention allows the electronic state of the fragment to emphasize different atoms or environments.

---

## 6. Returning Fragment Information to the Atoms

The fragment representation should then condition each atomic representation before the SCF parameters and atomic energies are evaluated.

For example,

[
\tilde h_i
==========

h_i^{\mathrm{geom}}
+
g_i\odot W f_F,
]

with

[
g_i
===

\sigma\left[
G(h_i^{\mathrm{geom}},f_F)
\right].
]

All atoms in a fragment therefore receive the same global fragment information, but its effect is atom dependent because it is combined with the local geometric representation.

This produces

[
\text{geometry}
\rightarrow
\text{atomic representation}
\rightarrow
\text{fragment electronic context}
\rightarrow
\text{fragment-conditioned atomic representation}.
]

The resulting (\tilde h_i) can be used to parameterize both the response functional and the charge-dependent atomic energy.

---

## 7. Feeding the SCF Electronic State Back into the Atomic Features

During the SCF procedure, the current multipoles and electrostatic environment can be incorporated into the atomic representation.

Because the electronic quantities have natural tensor ranks,

[
q_i \sim l=0,
\qquad
\boldsymbol{\mu}_i \sim l=1,
\qquad
\mathbf Q_i \sim l=2,
]

they can be inserted directly into matching equivariant feature channels.

Similarly,

[
V_i \sim l=0,
\qquad
\mathbf E_i \sim l=1,
\qquad
\nabla\mathbf E_i
\sim l=0\oplus l=2
]

after the appropriate irreducible decomposition.

The electronic-state-dependent feature can therefore be written schematically as

[
h_i^{\mathrm{SCF}}
==================

\mathrm{ElectronicUpdate}
\left(
\tilde h_i,
M_i,
V_i,
\mathbf E_i,
\nabla\mathbf E_i
\right).
]

The atomic energy is then

[
E_i^{\mathrm{atom}}
===================

E_\theta(h_i^{\mathrm{SCF}}).
]

Because this energy is part of the SCF functional itself,

[
E_{\mathrm{SCF}}
================

E_{\mathrm{FF}}(M)
+
\sum_i
E_\theta
\left(
\tilde h_i,
M_i,
V_i,
\mathbf E_i,
\nabla\mathbf E_i
\right),
]

changing the multipoles changes the atomic energies, while the resulting change in atomic energy changes the stationary multipoles.

The conceptual loop is therefore

[
M
\rightarrow
(V,\mathbf E,\nabla\mathbf E)
\rightarrow
h^{\mathrm{SCF}}
\rightarrow
E_{\mathrm{atom}}
\rightarrow
\frac{\partial E_{\mathrm{SCF}}}{\partial M}
\rightarrow
M.
]

This is the central mechanism by which intermolecular electronic response becomes directly coupled to intramolecular bonding.

The final output of the SCF is simply the physically relevant total set of atomic multipoles,

[
M^*=
\left(
q^*,
\boldsymbol{\mu}^*,
\mathbf Q^*,
\ldots
\right),
]

without requiring the model to assign a physically fundamental distinction between permanent and induced components.
