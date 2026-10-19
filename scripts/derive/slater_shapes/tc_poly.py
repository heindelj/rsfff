"""Two-center Coulomb overlap complements for shaped Slater densities rho_k = N_k r^k e^{-a r} (k = 0..KMAX),
derived in a polynomial ring: symbols ia = 1/a, ib = 1/b, iR = 1/R, w = 1/(a^2 - b^2), Ea = e^{-aR}, Eb = e^{-bR}
with their own derivative rules, so nothing is ever put over a common denominator.

Output per (k, l), order n in (1, 3, 5, 7, 9):
  closed:  comp = Ea * sum_m R^m Ca_m + Eb * sum_m R^m Cb_m,  C polynomials in (a, b, ia, ib, w)  [unequal]
  series:  comp = e^{-u} sum_p eps^p S_p(u),  a = z(1+eps), b = z(1-eps), u = z R              [near-equal]
"""
import sympy as sp, pickle, sys, time
KMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 2
OUT = sys.argv[2] if len(sys.argv) > 2 else f'tcpoly_K{KMAX}.pkl'
PMAX = 14
a, b, ia, ib, w, R, iR, Ea, Eb, e, u = sp.symbols('a b ia ib w R iR Ea Eb epsilon u')
RULES = {
    a: {a: 1, ia: -ia**2, w: -2*a*w**2, Ea: -R*Ea},
    b: {b: 1, ib: -ib**2, w: 2*b*w**2, Eb: -R*Eb},
    R: {R: 1, iR: -iR**2, Ea: -a*Ea, Eb: -b*Eb},
}


def D(expr, var, n=1):
    for _ in range(n):
        expr = sp.expand(sum(sp.diff(expr, s)*ds for s, ds in RULES[var].items()))
    return expr


def norm(expr):
    """apply a*ia = 1, b*ib = 1, R*iR = 1."""
    expr = sp.expand(expr)
    out = 0
    for t in sp.Add.make_args(expr):
        pw = t.as_powers_dict()
        for x, ix in ((a, ia), (b, ib), (R, iR)):
            p, q = pw.get(x, 0), pw.get(ix, 0)
            m = min(p, q)
            if m:
                t = t / (x**m * ix**m)
        out += t
    return sp.expand(out)


A = -b**2*w
B = a**2*w
f1_00 = 1 - A**2*(1 + 2*B + a*R/2)*Ea - B**2*(1 + 2*A + b*R/2)*Eb
I = sp.expand((1 - f1_00)*iR*(8*sp.pi)**2*ia**3*ib**3)
N = lambda k, x: x**(k+3)/(4*sp.pi*sp.factorial(k+2))

res = {}
for k in range(KMAX+1):
    for l in range(k, KMAX+1):
        t0 = time.time()
        J = norm(N(k, a)*N(l, b)*(-1)**(k+l)*D(D(I, a, k), b, l))
        F = {1: norm(1 - R*J)}
        for n in (1, 3, 5, 7):
            F[n+2] = norm(F[n] - R*D(F[n], R)/n)
        out = {}
        for n, f in F.items():
            comp = norm(1 - f)
            ca = sp.expand(comp.coeff(Ea, 1)); cb = sp.expand(comp.coeff(Eb, 1))
            assert norm(comp - ca*Ea - cb*Eb) == 0
            def bypow(c):
                d = {}
                for tt in sp.Add.make_args(c):
                    pw = tt.as_powers_dict()
                    m = int(pw.get(R, 0)) - int(pw.get(iR, 0))
                    d[m] = d.get(m, 0) + tt / (R**pw.get(R, 0) * iR**pw.get(iR, 0))
                return {m: sp.expand(v) for m, v in d.items()}
            Ca, Cb = bypow(ca), bypow(cb)
            # series: z = 1, a = 1+e, b = 1-e, ia = 1/(1+e), ib = 1/(1-e), w = 1/(4e), R = u
            geo = lambda s: sum((-s*e)**j for j in range(PMAX + 16))
            sub = {a: 1 + e, b: 1 - e, ia: geo(1), ib: geo(-1), w: 1/(4*e)}
            S = 0
            for C, sgn in ((Ca, -1), (Cb, 1)):
                ex = sum((sgn*u*e)**j/sp.factorial(j) for j in range(PMAX + 16))
                for m, c in C.items():
                    S += sp.expand(c.subs(sub)) * u**m * ex
            S = sp.expand(S)
            ser = []
            for p in range(-16, PMAX):
                c = sp.expand(S.coeff(e, p))
                if p < 0:
                    assert c == 0, (k, l, n, p)
                    continue
                pc = sp.Poly(c, u) if c != 0 else None
                if pc is not None:
                    assert all(m[0] >= 0 for m in pc.monoms())
                ser.append([str(x) for x in pc.all_coeffs()[::-1]] if pc is not None else ['0'])
            out[n] = {"Ca": {m: str(c) for m, c in Ca.items()}, "Cb": {m: str(c) for m, c in Cb.items()}, "series": ser}
        res[(k, l)] = out
        print(f"({k},{l}) {time.time()-t0:.0f}s; comp1 eps^0: {out[1]['series'][0]}", flush=True)
        pickle.dump(res, open(OUT, 'wb'))
