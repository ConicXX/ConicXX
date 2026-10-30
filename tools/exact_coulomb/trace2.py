"""Efficient pseudo-arclength path following of the exact De Saxce central path (dense, no smoothing):
per-cone relative metric, loose corrector (neighbourhood of the path), adaptive step aiming at ~3
corrector iterations. Counts linear solves. usage: trace2.py dump.txt [mu_end] [corr_tol]"""
import sys
import numpy as np
import dumpio

d = dumpio.read(sys.argv[1])
mu_end = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-12
ctol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-2
P, A, q, b = d["P"].toarray(), d["A"].toarray(), d["q"], d["b"]
n, m = A.shape[1], A.shape[0]
nc = m // 3
N = n + 2 * m
e = np.tile([1.0, 0, 0], nc)
solves = 0


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


def F(y, t):
    x, s, z = split(y)
    S = s.reshape(-1, 3)
    sh = np.zeros(m); sh[0::3] = np.linalg.norm(S[:, 1:], axis=1)
    return np.r_[P @ x + q + A.T @ z, A @ x + s - b - sh, jordan(s, z) - np.exp(t) * e]


def J(y, t):
    x, s, z = split(y)
    S = s.reshape(-1, 3)
    ell = np.linalg.norm(S[:, 1:], axis=1)
    L = np.zeros((m, m))
    for i in range(nc):
        if ell[i] > 0:
            L[3*i, 3*i+1:3*i+3] = S[i, 1:] / ell[i]
    Jy = np.zeros((N, N))
    Jy[:n, :n] = P; Jy[:n, n+m:] = A.T
    Jy[n:n+m, :n] = A; Jy[n:n+m, n:n+m] = np.eye(m) - L
    Jy[n+m:, n:n+m] = arw(z); Jy[n+m:, n+m:] = arw(s)
    return Jy, np.r_[np.zeros(n + m), -np.exp(t) * e]


def margin(y):
    _, s, z = split(y)
    S, Z = s.reshape(-1, 3), z.reshape(-1, 3)
    return min((S[:, 0] - np.linalg.norm(S[:, 1:], axis=1)).min(), (Z[:, 0] - np.linalg.norm(Z[:, 1:], axis=1)).min())


def max_step(y, dy, frac=0.9):
    if margin(y + dy) > 0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if margin(y + mid * dy) > 0: lo = mid
        else: hi = mid
    return frac * lo


def weights(y):
    x, s, z = split(y)
    w = np.empty(N + 1)
    w[:n] = 1 / max(np.abs(x).max(), 1e-300)
    w[n:n+m] = np.repeat(1 / np.maximum(np.linalg.norm(s.reshape(-1, 3), axis=1), 1e-300), 3)
    w[n+m:N] = np.repeat(1 / np.maximum(np.linalg.norm(z.reshape(-1, 3), axis=1), 1e-300), 3)
    w[N] = 1.0  # t = log mu is already relative
    return w


def solve(Jy, Jt, row, rhs):
    global solves
    solves += 1
    B = np.zeros((N + 1, N + 1)); B[:N, :N] = Jy; B[:N, N] = Jt; B[N, :] = row
    return np.linalg.solve(B, rhs)


t = np.log(1.0)
y = np.r_[np.zeros(n), e, e]
for k in range(100):  # damped Newton onto the path at mu = 1
    Jy, _ = J(y, t); r = F(y, t)
    if np.linalg.norm(r) < 1e-10: break
    dy = np.linalg.solve(Jy, -r); solves += 1
    y = y + max_step(y, dy, 0.95) * dy
Jy, Jt = J(y, t)
w = weights(y)
tan = solve(Jy, Jt, np.r_[np.zeros(N), 1.0], np.r_[np.zeros(N), -1.0])
h, steps, rejects = 0.3, 0, 0
while np.exp(t) > mu_end and steps < 3000:
    w = weights(y)
    tan /= np.linalg.norm(w * tan)
    Y, T = y + h * tan[:N], t + h * tan[N]
    a = max_step(y, h * tan[:N])
    if a < 1:  # the predictor leaves the cone: shorten the step itself
        h *= a
        Y, T = y + h * tan[:N], t + h * tan[N]
    ok = False
    for it in range(1, 8):
        Jy, Jt = J(Y, T)
        dY = solve(Jy, Jt, w**2 * tan, np.r_[-F(Y, T), 0.0])
        al = max_step(Y, dY[:N])
        Y, T = Y + al * dY[:N], T + al * dY[N]
        if al == 1.0 and np.linalg.norm(weights(Y) * dY) < ctol:
            ok = True
            break
    if not ok:
        h *= 0.4; rejects += 1
        if h < 1e-8: print("step size collapsed"); break
        continue
    Jy, Jt = J(Y, T)
    new = solve(Jy, Jt, np.r_[w**2 * tan], np.r_[np.zeros(N), 1.0])
    if new[N] * tan[N] < 0: print(f"  fold at mu {np.exp(T):.2e}")
    y, t, tan = Y, T, new
    steps += 1
    h = min(h * (2.0 if it <= 2 else 1.3 if it == 3 else 1.0 if it <= 5 else 0.6), 10.0)
    if steps % 10 == 0:
        print(f"step {steps:4d}: mu {np.exp(t):.2e} h {h:.2e} corr {it} solves {solves}")
x, s, z = split(y)
print(f"end: mu {np.exp(t):.2e}, {steps} steps, {rejects} rejected, {solves} linear solves, objective {0.5 * x @ P @ x + q @ x:.8e}")
np.savetxt(sys.argv[1] + ".trace_end", np.c_[np.r_[x, np.zeros(m - n)], s, z])
