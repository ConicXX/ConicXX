"""Read/write the CARDILLO_CONICXX_DUMP format (P, q, A, b, compliance, cone spec)."""
import numpy as np
import scipy.sparse as sp


def read(path):
    tok = open(path).read().split()
    pos = 0

    def nxt(n=1):
        nonlocal pos
        v = tok[pos:pos + n]
        pos += n
        return v

    def mat():
        r, c, nnz = map(int, nxt(3))
        d = np.array(nxt(3 * nnz), dtype=float).reshape(nnz, 3) if nnz else np.zeros((0, 3))
        return sp.csc_matrix((d[:, 2], (d[:, 0].astype(int), d[:, 1].astype(int))), shape=(r, c))

    def vec():
        n = int(nxt()[0])
        return np.array(nxt(n), dtype=float)

    P = mat(); q = vec(); A = mat(); b = vec(); comp = vec()
    zero, nonneg, nsoc, coul = map(int, nxt(4))
    return dict(P=P, q=q, A=A, b=b, comp=comp, zero=zero, nonneg=nonneg, nsoc=nsoc, coul=coul)


def write(path, d):
    with open(path, "w") as f:
        def mat(M):
            M = M.tocoo()
            f.write(f"{M.shape[0]} {M.shape[1]} {M.nnz}\n")
            for i, j, v in zip(M.row, M.col, M.data):
                f.write(f"{i} {j} {v:.17g}\n")

        def vec(v):
            f.write(f"{len(v)}\n")
            f.writelines(f"{x:.17g}\n" for x in v)

        mat(d["P"]); vec(d["q"]); mat(d["A"]); vec(d["b"]); vec(d["comp"])
        f.write(f"{d['zero']} {d['nonneg']} {d['nsoc']} {d['coul']}\n")
