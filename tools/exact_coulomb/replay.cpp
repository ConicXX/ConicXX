// Replays a ConicXX problem dumped by CardilloCxx (CARDILLO_CONICXX_DUMP) with CardilloCxx-like settings.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include "dumpread.h"
using namespace conicxx;
int main(int argc, char** argv) {
  Dump d = readDump(argv[1]);
  const SparseMat &P = d.P, &A = d.A;
  const Vec &q = d.q, &b = d.b, &comp = d.comp;
  ConeSpec spec = d.spec;
  const long nsoc = static_cast<long>(spec.soc_dims.size());
  const bool exact = argc > 2 ? std::atoi(argv[2]) : d.coulomb;
  if (exact) spec.soc_coulomb.assign(nsoc, 1);
  Settings s;
  const double tol = argc > 3 ? std::atof(argv[3]) : 1e-6;
  s.tol_feas = s.tol_gap_abs = s.tol_gap_rel = s.tol_infeas_abs = s.tol_infeas_rel = tol;
  s.max_iter = 100;
  s.regularization.static_P = s.regularization.static_nonneg = s.regularization.static_soc = s.regularization.static_zero = 1e-8;
  s.regularization.dynamic_delta = 1e-8;
  s.refine_max_iter = 30;
  s.verbose = argc > 4 ? std::atoi(argv[4]) : 2;
  if (argc > 5) s.coulomb.smoothing = std::atof(argv[5]);
  if (argc > 6) s.coulomb.mu_floor = std::atof(argv[6]);
  if (const char* th = std::getenv("XP_THETA")) s.centrality_theta = std::atof(th);
  Solver solver;
  if (!solver.setup(P, q, A, b, spec, s, &comp)) { std::printf("setup failed\n"); return 1; }
  const Solution& sol = solver.solve();
  if (argc > 7) {  // x, s, z (x padded) for subproblem extraction and comparisons
    std::ofstream o(argv[7]);
    o.precision(17);
    for (Index i = 0; i < sol.s.size(); ++i) o << (i < sol.x.size() ? sol.x[i] : 0.0) << " " << sol.s[i] << " " << sol.z[i] << "\n";
  }
  std::printf("status %s, %d iterations, %.1f ms\n", toString(sol.status), sol.info.iterations, 1e3 * sol.info.solve_time_s);
  return 0;
}
