// Quickstart: one problem that uses all three cone types.
//
// Variables x = (x0, x1, x2). Minimize 0.5*||x - c||^2 with target c = (1, 1, 0.3), subject to
//   x0 + x1 = 1               equality     -> zero cone,         1 row
//   x2 >= 0                   inequality   -> nonnegative cone,  1 row
//   x2 >= ||(x0, x1)||_2      norm bound   -> second-order cone, 3 rows
// The rows of A and b must appear in the order zero -> nonnegative -> second-order.

#include <conicxx/solver.h>

#include <cstdio>
#include <vector>

int main() {
  using namespace conicxx;

  // P = I (upper triangle only), q = -c.
  SparseMat P(3, 3);
  {
    std::vector<Triplet> t = {{0, 0, 1.0}, {1, 1, 1.0}, {2, 2, 1.0}};
    P.setFromTriplets(t.begin(), t.end());
  }
  Vec q(3);
  q << -1.0, -1.0, -0.3;

  // s = b - A x must lie in K:
  //   row 0 (zero):        s0 = 1 - x0 - x1        = 0
  //   row 1 (nonnegative): s1 = x2                 >= 0
  //   rows 2-4 (SOC):      (s2, s3, s4) = (x2, x0, x1) in Q^3
  SparseMat A(5, 3);
  {
    std::vector<Triplet> t = {{0, 0, 1.0},  {0, 1, 1.0},  {1, 2, -1.0},
                              {2, 2, -1.0}, {3, 0, -1.0}, {4, 1, -1.0}};
    A.setFromTriplets(t.begin(), t.end());
  }
  Vec b(5);
  b << 1.0, 0.0, 0.0, 0.0, 0.0;

  ConeSpec cones;
  cones.zero_dim = 1;
  cones.nonneg_dim = 1;
  cones.soc_dims = {3};

  Settings settings;
  settings.verbose = 2;  // one line per iteration

  Solver solver;
  if (!solver.setup(P, q, A, b, cones, settings)) {
    std::fprintf(stderr, "setup() rejected the problem data\n");
    return 1;
  }
  const Solution& sol = solver.solve();

  // Expected: x = (0.5, 0.5, 0.70710678): the equality and symmetry give x0 = x1 = 0.5, and the
  // second-order cone lifts x2 from its target 0.3 to sqrt(0.5^2 + 0.5^2).
  std::printf("status    = %s\n", toString(sol.status));
  std::printf("x         = (%.8f, %.8f, %.8f)\n", sol.x[0], sol.x[1], sol.x[2]);
  std::printf("objective = %.8f\n", sol.objective);
  std::printf("iters     = %d\n", sol.info.iterations);
  return sol.ok() ? 0 : 1;
}
