"""Exact central path of the De Saxce SOCCP (all cones 3-dim Coulomb, smoothing eps = c*mu) by
pseudo-arclength continuation in (x, s, z, log mu), dense. Reports folds (sign changes of dmu/dsigma)
and the det sign of the Jacobian. usage: trace.py dump.txt [c] [out.npz]"""
import sys
import numpy as np
import dumpio

d = dumpio.read(sys.argv[1])
c = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
P, A, q, b = d["P"].toarray(), d["A"].toarray(), d["q"], d["b"]
n, m = A.shape[1], A.shape[0]
nc = m // 3
N = n + 2 * m


def split(y):
    return y[:n], y[n:n + m], y[n + m:]


def arw(v):
    V = v.reshape(-1, 3)
    M = np.zeros((m, m))
    for i, (v0, v1, v2) in enumerate(V):
        M[3*i:3*i+3, 3*i:3*i+3] = [[v0, v1, v2], [v1, v0, 0], [v2, 0, v0]]
    return M


def jordan(s, z):
    S, Z = s.reshape(-1, 3), z.reshape(-1, 3)
    out = np.empty_like(S)
    out[:, 0] = (S * Z).sum(1)
    out[:, 1:] = S[:, :1] * Z[:, 1:] + Z[:, :1] * S[:, 1:]
    return out.ravel()


e = np.tile([1.0, 0, 0], nc)


def F(y, t):
    x, s, z = split(y)
    mu = np.exp(t)
    S = s.reshape(-1, 3)
    ell = np.sqrt((S[:, 1:]**2).sum(1) + (c * mu)**2)
    sh = np.zeros(m); sh[0::3] = ell
    return np.r_[P @ x + q + A.T @ z, A @ x + s - b - sh, jordan(s, z) - mu * e]


def J(y, t):
    x, s, z = split(y)
    mu = np.exp(t)
    S = s.reshape(-1, 3)
    ell = np.sqrt((S[:, 1:]**2).sum(1) + (c * mu)**2)
    L = np.zeros((m, m))
    for i in range(nc):
        L[3*i, 3*i+1:3*i+3] = S[i, 1:] / ell[i] if ell[i] > 0 else 0
    Jy = np.zeros((N, N))
    Jy[:n, :n] = P; Jy[:n, n+m:] = A.T
    Jy[n:n+m, :n] = A; Jy[n:n+m, n:n+m] = np.eye(m) - L
    Jy[n+m:, n:n+m] = arw(z); Jy[n+m:, n+m:] = arw(s)
    dsh = np.zeros(m); dsh[0::3] = (c * mu)**2 / np.maximum(ell, 1e-300)  # d ell / d t
    Jt = np.r_[np.zeros(n), -dsh, -mu * e]
    return Jy, Jt


def interior(y, frac=0.0):
    _, s, z = split(y)
    S, Z = s.reshape(-1, 3), z.reshape(-1, 3)
    return (S[:, 0] - np.linalg.norm(S[:, 1:], axis=1)).min() > 0 and (Z[:, 0] - np.linalg.norm(Z[:, 1:], axis=1)).min() > 0


def max_step(y, dy):
    a = 1.0
    while not interior(y + a * dy) and a > 1e-12:
        a *= 0.5
    return a


# starting point on the path at mu0 by damped Newton
t = np.log(10.0)
y = np.r_[np.zeros(n), 10 * e, e]
for k in range(200):
    r = F(y, t)
    if np.linalg.norm(r) < 1e-12 * (1 + np.linalg.norm(y)):
        break
    Jy, _ = J(y, t)
    dy = np.linalg.solve(Jy, -r)
    y = y + 0.95 * max_step(y, dy) * dy
print(f"start mu {np.exp(t):.2e}, |F| {np.linalg.norm(F(y, t)):.1e} after {k} Newton steps")

# weights: arclength in relative units of the current point
def W(y):
    return 1.0 / (np.abs(y) + 1e-8 * np.abs(y).max())

Jy, Jt = J(y, t)
ty = np.linalg.solve(Jy, -Jt)
tan = -np.r_[ty, 1.0]  # towards decreasing mu
h = 0.05
hist = []
sign_prev = np.sign(np.linalg.slogdet(Jy)[0])
x_ref = None
while np.exp(t) > 1e-14 and len(hist) < 4000:
    w = np.r_[W(y), 1.0]
    tan = tan / np.sqrt(np.sum((w * tan)**2))
    # predictor, shortened to stay interior
    a = max_step(y, h * tan[:N])
    hh = min(h, 0.9 * a * h) if a < 1 else h
    Y, T = y + hh * tan[:N], t + hh * tan[N]
    ok = False
    for it in range(15):
        Jy, Jt = J(Y, T)
        r = F(Y, T)
        if np.linalg.norm(r) < 1e-10 * (1 + np.abs(Y).max()):
            ok = True
            break
        # bordered Newton: [Jy Jt; (w^2 tan)'] dY = [-r; 0]
        B = np.zeros((N + 1, N + 1))
        B[:N, :N] = Jy; B[:N, N] = Jt; B[N, :] = w**2 * tan
        dY = np.linalg.solve(B, np.r_[-r, 0.0])
        al = max_step(Y, dY[:N])
        if al < 1:
            al *= 0.9
        Y, T = Y + al * dY[:N], T + al * dY[N]
    if not ok:
        h *= 0.5
        if h < 1e-10:
            print("step size collapsed"); break
        continue
    Jy, Jt = J(Y, T)
    B = np.zeros((N + 1, N + 1))
    B[:N, :N] = Jy; B[:N, N] = Jt; B[N, :] = w**2 * tan
    new = np.linalg.solve(B, np.r_[np.zeros(N), 1.0])  # tangent with (w^2 tan)' new = 1 keeps the orientation
    sg, _ = np.linalg.slogdet(Jy)
    if np.sign(new[N]) != np.sign(tan[N]):
        print(f"  FOLD near mu {np.exp(T):.3e}: dt/dsigma changes sign ({tan[N]:+.2e} -> {new[N]:+.2e})")
    if sg != sign_prev:
        print(f"  det(J_y) changes sign near mu {np.exp(T):.3e}")
        sign_prev = sg
    y, t, tan = Y, T, new
    x_, s_, z_ = split(y)
    hist.append((np.exp(t), tan[N] / np.sqrt(np.sum((w * tan)**2)), x_.copy(), s_.copy(), z_.copy()))
    if len(hist) % 25 == 0:
        print(f"step {len(hist):4d}: mu {np.exp(t):.3e}  h {h:.2e}  dlogmu/dsigma {tan[N]:+.2e}  obj {0.5 * x_ @ P @ x_ + q @ x_:.8e}")
    h = min(1.5 * h, 0.5) if it < 4 else h * 0.7
x, s, z = split(y)
print(f"end mu {np.exp(t):.3e}, objective {0.5 * x @ P @ x + q @ x:.8e}")
if len(sys.argv) > 3:
    np.savez(sys.argv[3], mu=np.array([h_[0] for h_ in hist]), x=np.array([h_[2] for h_ in hist]),
             s=np.array([h_[3] for h_ in hist]), z=np.array([h_[4] for h_ in hist]))
    np.savetxt(sys.argv[3] + ".end", np.c_[np.r_[x, np.zeros(m - n)], s, z])
