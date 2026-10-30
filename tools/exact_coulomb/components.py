"""Connected components of the body/contact graph of a dump; writes the components containing given cones.
usage: components.py dump.txt out_prefix cone [cone ...]"""
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cg
import dumpio

d = dumpio.read(sys.argv[1])
P, A = d["P"], d["A"]
assert d["zero"] == 0 and d["nonneg"] == 0
n = P.shape[0]
cone_of_row = np.arange(A.shape[0]) // 3
# variable graph: P couplings and rows of the same cone
Ac = sp.csr_matrix((np.ones(A.nnz), (cone_of_row[A.tocoo().row], A.tocoo().col)), shape=(d["nsoc"], n))
G = (abs(P) + Ac.T @ Ac) != 0
ncomp, lab = cg.connected_components(G, directed=False)
sizes = np.bincount(lab)
print(f"{ncomp} components, sizes: max {sizes.max()}, hist {np.bincount(np.minimum(sizes, 60))[[6,12,18,24,30,36,42,48,54]]} (6..54 dofs)")
for c in map(int, sys.argv[3:]):
    vars_ = np.unique(Ac[c].indices)
    k = lab[vars_[0]]
    cols = np.flatnonzero(lab == k)
    cones = np.flatnonzero(np.asarray((Ac[:, cols] != 0).sum(axis=1)).ravel() > 0)
    rows = (3 * cones[:, None] + np.arange(3)).ravel()
    sub = dict(P=P[cols][:, cols], q=d["q"][cols], A=A[rows][:, cols], b=d["b"][rows], comp=d["comp"][rows],
               zero=0, nonneg=0, nsoc=len(cones), coul=d["coul"])
    out = f"{sys.argv[2]}_c{c}.txt"
    dumpio.write(out, sub)
    print(f"cone {c}: component {k}: {len(cols)} dofs, {len(cones)} cones, local index {int(np.searchsorted(cones, c))} -> {out}")
