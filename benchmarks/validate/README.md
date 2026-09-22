# M6 — is the torchff backend the same force field?

`validate_backends.py` evaluates a trained FilmModel checkpoint on both backends and reports

* **parity** on `benchmarks/structures` (w4–w23): energy, forces, the four interaction
  components, and on every k-th structure the force-loss gradient into every network
  parameter (the double backward training runs on) — largest absolute and relative
  differences per quantity;
* **NVE** on a 216-water cluster (velocity Verlet in the script itself, on the device):
  total-energy drift and RMS relative to the kinetic energy on torchff, a short rerun on the
  torch backend from the same start, and the step-by-step difference between the two.

```bash
bash benchmarks/validate/run_validate.sh                 # GPU node, ~15 min
python benchmarks/validate/validate_backends.py --device cpu --skip-nve \
    --structures benchmarks/structures/w4_mp2_avtz.xyz   # Mac smoke test
```

Results land in `results/<tag>_<stamp>.md` + `.json` (the JSON keeps the full NVE log).

What "pass" looks like: relative differences at round-off (1e-13 or better on energies and
components, 1e-10 or better on forces and the parameter gradient — the kernels sum in a
different order and use atomics), and an NVE run whose `E_tot RMS / <E_kin>` is well below
1e-2 with no drift that the torch backend does not also show over its comparison stretch.
On CPU the torchff backend runs torchff-lib's pure-torch references, so a CPU run checks the
wiring, not the kernels.
