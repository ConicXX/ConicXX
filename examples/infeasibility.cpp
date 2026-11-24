// Infeasibility detection and certificates.
//
// 1. A primal infeasible LP: x >= 1 and x <= 0.
//    Certificate: z in K* with A'z = 0 and b'z < 0 (normalized to b'z = -1).
// 2. A dual infeasible (unbounded) LP: minimize -x subject to x >= 0.
//    Certificate: x with Ax + s = 0 for some s in K, Px = 0 and q'x < 0 (normalized to q'x = -1).

#include <conicxx/solver.h>

#include <cstdio>
#include <vector>

int main() {
  using namespace conicxx;

  // --- 1. primal infeasible ---
  {
    SparseMat P(1, 1);  // P = 0
    Vec q(1);
    q << 1.0;
    // s0 = -1 - (-x) = x - 1 >= 0  and  s1 = 0 - x >= 0
    SparseMat A(2, 1);
    std::vector<Triplet> t = {{0, 0, -1.0}, {1, 0, 1.0}};
    A.setFromTriplets(t.begin(), t.end());
    Vec b(2);
    b << -1.0, 0.0;
    ConeSpec cones;
    cones.nonneg_dim = 2;

    Solver solver;
    if (!solver.setup(P, q, A, b, cones)) return 1;
    const Solution& sol = solver.solve();
    const Scalar Atz = sol.z[0] * -1.0 + sol.z[1] * 1.0;
    std::printf("[primal] status = %s, z = (%.6f, %.6f), A'z = %.2e, b'z = %.6f\n",
                toString(sol.status), sol.z[0], sol.z[1], Atz, b.dot(sol.z));
    if (sol.status != Status::PrimalInfeasible) return 1;
  }

  // --- 2. dual infeasible ---
  {
    SparseMat P(1, 1);  // P = 0
    Vec q(1);
    q << -1.0;
    SparseMat A(1, 1);  // s = 0 - (-x) = x >= 0
    std::vector<Triplet> t = {{0, 0, -1.0}};
    A.setFromTriplets(t.begin(), t.end());
    Vec b(1);
    b << 0.0;
    ConeSpec cones;
    cones.nonneg_dim = 1;

    Solver solver;
    if (!solver.setup(P, q, A, b, cones)) return 1;
    const Solution& sol = solver.solve();
    std::printf("[dual]   status = %s, x = %.6f, s = %.6f, q'x = %.6f\n", toString(sol.status),
                sol.x[0], sol.s[0], q.dot(sol.x));
    if (sol.status != Status::DualInfeasible) return 1;
  }
  return 0;
}
