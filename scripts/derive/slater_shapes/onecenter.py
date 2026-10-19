import sympy as sp
r, s, z, u = sp.symbols('r s zeta u', positive=True)
out = {}
for k in range(4):
    N = z**(k+3)/(4*sp.pi*sp.factorial(k+2))
    rho = N*s**k*sp.exp(-z*s)
    assert sp.simplify(sp.integrate(rho*4*sp.pi*s**2, (s, 0, sp.oo)) - 1) == 0
    phi = sp.integrate(rho*4*sp.pi*s**2, (s, 0, r))/r + sp.integrate(rho*4*sp.pi*s, (s, r, sp.oo))
    f1 = sp.simplify(phi*r)                         # = 1 - e^{-u} P(u)
    F = {1: f1}
    for n in (1, 3):
        F[n+2] = sp.simplify(F[n] - r*sp.diff(F[n], r)/n)
    polys = {}
    for n, f in F.items():
        P = sp.expand(sp.simplify((1 - f)*sp.exp(z*r)).subs(r, u/z))
        polys[n] = [sp.Rational(c) for c in sp.Poly(P, u).all_coeffs()[::-1]]
    out[k] = polys
    print(k, {n: [str(c) for c in p] for n, p in polys.items()})
import pickle; pickle.dump({k: {n: [str(c) for c in p] for n, p in v.items()} for k, v in out.items()}, open('onecenter.pkl', 'wb'))
