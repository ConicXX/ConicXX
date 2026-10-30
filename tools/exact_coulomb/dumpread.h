// Reader of the problems dumped by CardilloCxx (CARDILLO_CONICXX_DUMP).
#pragma once
#include <fstream>
#include <vector>
#include "conicxx/solver.h"
struct Dump {
  conicxx::SparseMat P, A;
  conicxx::Vec q, b, comp;
  conicxx::ConeSpec spec;
  bool coulomb = false;
};
inline Dump readDump(const char* path) {
  using namespace conicxx;
  std::ifstream f(path);
  auto mat = [&] {
    long r, c, nnz;
    f >> r >> c >> nnz;
    std::vector<Triplet> t;
    t.reserve(nnz);
    for (long k = 0; k < nnz; ++k) {
      long i, j;
      double v;
      f >> i >> j >> v;
      t.emplace_back(i, j, v);
    }
    SparseMat M(r, c);
    M.setFromTriplets(t.begin(), t.end());
    M.makeCompressed();
    return M;
  };
  auto vec = [&] {
    long n;
    f >> n;
    Vec v(n);
    for (long i = 0; i < n; ++i) f >> v[i];
    return v;
  };
  Dump d;
  d.P = mat();
  d.q = vec();
  d.A = mat();
  d.b = vec();
  d.comp = vec();
  long nsoc;
  int coul;
  f >> d.spec.zero_dim >> d.spec.nonneg_dim >> nsoc >> coul;
  d.spec.soc_dims.assign(nsoc, 3);
  d.coulomb = coul;
  return d;
}
