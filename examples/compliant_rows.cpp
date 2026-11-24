// Compliant equality rows.
//
//   minimize 0.5*(x0^2 + x1^2)  subject to  x0 - x1 = 1  (compliance c)
//
// With compliance c >= 0 the row becomes x0 - x1 - c*z = 1, where z is the row's multiplier.
// This is the same as adding a variable pi = -z with cost 0.5*c*pi^2 and the row
// x0 - x1 + c*pi = 1: a spring of compliance c (stiffness 1/c) in series with the constraint.
// The exact solution is x0 = -x1 = 1/(2 + c); c = 0 is the rigid equality.

#include <conicxx/solver.h>

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
  using namespace conicxx;

  SparseMat P(2, 2);
  {
    std::vector<Triplet> t = {{0, 0, 1.0}, {1, 1, 1.0}};
    P.setFromTriplets(t.begin(), t.end());
  }
  Vec q = Vec::Zero(2);
  SparseMat A(1, 2);  // s = 1 - (x0 - x1) = 0
  {
    std::vector<Triplet> t = {{0, 0, 1.0}, {0, 1, -1.0}};
    A.setFromTriplets(t.begin(), t.end());
  }
  Vec b(1);
  b << 1.0;
  ConeSpec cones;
  cones.zero_dim = 1;

  Vec c(1);
  c << 0.0;
  Solver solver;
  if (!solver.setup(P, q, A, b, cones, Settings{}, &c)) return 1;

  for (const Scalar compliance : {0.0, 0.5, 2.0, 1e3}) {
    c << compliance;
    solver.updateData(nullptr, nullptr, nullptr, nullptr, &c);  // only the compliance changes
    const Solution& sol = solver.solve();
    const Scalar exact = 1.0 / (2.0 + compliance);
    std::printf("c = %8.1e: status = %s, x = (%+.8f, %+.8f), exact x0 = %.8f, z = %+.8f\n",
                compliance, toString(sol.status), sol.x[0], sol.x[1], exact, sol.z[0]);
    if (!sol.ok() || std::abs(sol.x[0] - exact) > 1e-6) return 1;
  }
  return 0;
}
