import sympy as sp
r, s, z, u = sp.symbols('r s zeta u', positive=True)
for k in range(4):
    N = z**(k+3)/(4*sp.pi*sp.factorial(k+2)); rho = N*s**k*sp.exp(-z*s)
    f = {1: sp.simplify(r*(sp.integrate(rho*4*sp.pi*s**2, (s, 0, r))/r + sp.integrate(rho*4*sp.pi*s, (s, r, sp.oo))))}
    for n in (1, 3, 5, 7):
        f[n+2] = sp.simplify(f[n] - r*sp.diff(f[n], r)/n)
    print(k, {n: [str(c) for c in sp.Poly(sp.expand(sp.simplify((1-f[n])*sp.exp(z*r)).subs(r, u/z)), u).all_coeffs()[::-1]] for n in (7, 9)})
