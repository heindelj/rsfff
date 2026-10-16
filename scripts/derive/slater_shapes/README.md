# Shaped Slater density factors (sympy derivations)

Generates `src/ff/_slater_shape_tables.py` (two-center) and the one-center polynomials in
`src/ff/multipole.py` (`SHAPE_ONE_CENTER`) for densities rho_k = N_k r^k e^{-b r}, k = 0..3.

    python tc_poly.py 3 tcpoly_K3.pkl     # ~90 s: closed forms + near-equal series (to eps^13)
    python codegen.py                      # -> _slater_shape_tables.py (copy into src/ff/)
    python onecenter.py; python onecenter9.py   # one-center polynomials (orders 1-9)

Method: two-center integrals J_kl = N_k N_l (-d/da)^k (-d/db)^l I(a, b, R) from the 1s-1s Coulomb integral
(HIPPO, Rackers et al. 2021, App. A), carried out in a polynomial ring (1/a, 1/b, 1/R, 1/(a^2 - b^2) as
symbols) so nothing is put over a common denominator; higher orders by f_{n+2} = f_n - R f_n'/n. The
near-equal series substitutes a = 1 + eps, b = 1 - eps (the factors are dimensionless in u = zR).
Validation: tests/test_ff_multipole_shapes.py (quadrature, recursion, limits, branches).
