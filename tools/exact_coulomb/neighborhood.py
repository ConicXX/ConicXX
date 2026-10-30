"""Subproblem of the bodies within `hops` contact rings of a seed cone; the other bodies are fixed at a
reference x (moved into b and q). usage: neighborhood.py dump.txt x_ref.txt cone hops out.txt"""
import sys
import numpy as np
import scipy.sparse as sp
import dumpio

d = dumpio.read(sys.argv[1])
xref = np.loadtxt(sys.argv[2])
xref = xref[:, 0] if xref.ndim == 2 else xref
seed, hops, out = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
P, A = d["P"].tocsr(), d["A"].tocsr()
n, nc = P.shape[0], d["nsoc"]
body = np.arange(n) // 6
coo = A.tocoo()
Cb = sp.csr_matrix((np.ones(coo.nnz), (coo.row // 3, body[coo.col])), shape=(nc, n // 6)) != 0
bodies = set(Cb[seed].indices)
for _ in range(hops):
    cones = np.flatnonzero(np.asarray(Cb[:, sorted(bodies)].sum(axis=1)).ravel() > 0)
    bodies |= set(Cb[cones].indices)
bodies = np.array(sorted(bodies))
cols = (6 * bodies[:, None] + np.arange(6)).ravel()
out_cols = np.setdiff1d(np.arange(n), cols)
cones = np.flatnonzero(np.asarray(Cb[:, bodies].sum(axis=1)).ravel() > 0)
rows = (3 * cones[:, None] + np.arange(3)).ravel()
Ar = A[rows]
sub = dict(P=P[cols][:, cols], q=d["q"][cols] + P[cols][:, out_cols] @ xref[out_cols], A=Ar[:, cols],
           b=d["b"][rows] - Ar[:, out_cols] @ xref[out_cols], comp=d["comp"][rows] if len(d["comp"]) else d["comp"],
           zero=0, nonneg=0, nsoc=len(cones), coul=d["coul"])
dumpio.write(out, sub)
np.savetxt(out + ".map", np.r_[cones], fmt="%d")
print(f"seed {seed}, {hops} hops: {len(bodies)} bodies, {len(cones)} cones, seed local {int(np.searchsorted(cones, seed))}")
