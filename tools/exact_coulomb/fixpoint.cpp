// De Saxce fixed point on a dumped problem: convex solves with b + E ||s_bar||, until the shift settles.
// Writes x, s, z of the result (columns) to argv[3]. usage: fixpoint dump tol out [max_outer]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "dumpread.h"
using namespace conicxx;
int main(int argc, char** argv) {
  Dump d = readDump(argv[1]);
  const double tol = std::atof(argv[2]);
  const int max_outer = argc > 4 ? std::atoi(argv[4]) : 200;
  const Index nc = static_cast<Index>(d.spec.soc_dims.size()), off = d.spec.zero_dim + d.spec.nonneg_dim;
  Settings st;
  st.tol_feas = st.tol_gap_abs = st.tol_gap_rel = st.tol_infeas_abs = st.tol_infeas_rel = tol;
  st.max_iter = 200;
  st.verbose = 0;
  Vec ell = Vec::Zero(nc), b = d.b;
  Solver solver;
  Solution sol;
  for (int k = 0; k < max_outer; ++k) {
    for (Index i = 0; i < nc; ++i) b[off + 3 * i] = d.b[off + 3 * i] + ell[i];
    if (!solver.setup(d.P, d.q, d.A, b, d.spec, st, d.comp.size() ? &d.comp : nullptr)) return 1;
    sol = solver.solve();
    Scalar change = 0;
    for (Index i = 0; i < nc; ++i) {
      const Scalar l = sol.s.segment<2>(off + 3 * i + 1).norm();
      change = std::max(change, std::abs(l - ell[i]));
      ell[i] = l;
    }
    std::printf("outer %3d: %s, %d it, max |dl| %.3e\n", k, toString(sol.status), sol.info.iterations, change);
    if (change < tol) break;
  }
  FILE* o = std::fopen(argv[3], "w");
  for (Index i = 0; i < d.A.rows(); ++i) std::fprintf(o, "%.17g %.17g %.17g\n", i < sol.x.size() ? sol.x[i] : 0.0, sol.s[i], sol.z[i]);
  std::fclose(o);
  return 0;
}
