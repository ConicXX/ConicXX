"""Exact De Saxce KKT residuals of an (x, s, z) file: dual Px+q+A'z, primal Ax+s-b-E||s_bar||, cone
violations, per-cone complementarity in velocity units (|s o z| / |z|). usage: kkt.py dump.txt sol"""
import sys
import numpy as np
import dumpio

d = dumpio.read(sys.argv[1])
X = np.loadtxt(sys.argv[2])
n = d["P"].shape[0]
x, s, z = X[:n, 0], X[:, 1], X[:, 2]
S, Z = s.reshape(-1, 3), z.reshape(-1, 3)
shift = np.zeros_like(s)
shift[0::3] = np.linalg.norm(S[:, 1:], axis=1)
rd = d["P"] @ x + d["q"] + d["A"].T @ z
rp = d["A"] @ x + s - d["b"] - shift
cs = np.maximum(0, np.linalg.norm(S[:, 1:], axis=1) - S[:, 0]).max()
cz = np.maximum(0, np.linalg.norm(Z[:, 1:], axis=1) - Z[:, 0]).max()
sz0 = (S * Z).sum(1)
szb = S[:, 0, None] * Z[:, 1:] + Z[:, 0, None] * S[:, 1:]
comp = np.sqrt(sz0**2 + (szb**2).sum(1))
vel = comp / np.maximum(np.linalg.norm(Z, axis=1), 1e-300)
print(f"|rd| {abs(rd).max():.2e} (|q| {abs(d['q']).max():.2e})  |rp| {abs(rp).max():.2e}  cone viol s {cs:.1e} z {cz:.1e}")
print(f"complementarity: max |s o z| {comp.max():.2e}, max velocity error {vel.max():.2e} (cone {vel.argmax()}), sum s'z {sz0.sum():.2e}")
print(f"objective 1/2 x'Px + q'x = {0.5 * x @ (d['P'] @ x) + d['q'] @ x:.6e}")
