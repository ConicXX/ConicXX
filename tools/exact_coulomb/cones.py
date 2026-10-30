"""Per-cone table of a (x, s, z) file from fixpoint: velocity u = s - E||s_bar||, impulse z, state.
usage: cones.py dump.txt sol.fp [cone ...]"""
import sys
import numpy as np
import dumpio

d = dumpio.read(sys.argv[1])
X = np.loadtxt(sys.argv[2])
s, z = X[:, 1].reshape(-1, 3), X[:, 2].reshape(-1, 3)
sel = list(map(int, sys.argv[3:])) or range(len(s))
st = np.linalg.norm(s[:, 1:], axis=1)
zt = np.linalg.norm(z[:, 1:], axis=1)
un = s[:, 0] - st
zmax = z[:, 0].max()
print(f"max z_N {zmax:.3e}; cones {len(s)}")
def state(i):
    stick = st[i] < 1e-6 * max(1e-3, abs(s[i]).max())
    if z[i, 0] < 1e-6 * zmax: return "takeoff" + ("(grazing)" if un[i] < 1e-7 else "")
    if st[i] < 1e-7: return "stick" + ("(at limit)" if z[i, 0] - zt[i] < 1e-6 * z[i, 0] else "")
    return "slip"
for i in sel:
    print(f"{i:4d} u_N {un[i]: .2e} |u_T| {st[i]:.2e}  z_N {z[i,0]:.3e} |z_T|/z_N {zt[i]/max(z[i,0],1e-300):.6f}  cos {np.dot(s[i,1:],z[i,1:])/max(st[i]*zt[i],1e-300): .4f}  {state(i)}")
from collections import Counter
print(Counter(state(i) for i in range(len(s))))
