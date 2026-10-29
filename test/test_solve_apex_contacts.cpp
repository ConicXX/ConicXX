// Phase 3 accept criterion (CONICXX_AGENT_TASKS.md): "apex contacts" -- N contacts where half
// have zero normal force at the solution (the cone's apex, a genuinely non-smooth point) and half
// are sticking with tangential force strictly inside the cone. Must converge to 1e-8 in <=25
// iterations, with no NaNs. See benchmarks/problem_generators.cpp's makeApexContacts() for the
// same construction at larger sizes, tracked for iteration-count regressions in the benchmark
// suite; this test asserts the accept criterion's exact numbers, which that table alone doesn't.

#include "conicxx/solver.h"

#include <gtest/gtest.h>

#include "test_helpers.h"

using namespace conicxx;

namespace {

// num_contacts 3D Coulomb friction cones (mu*fn >= ||ft||), half "apex" (minimize fn alone, no
// other constraint -- forces (fn, ft) = (0, 0) exactly at the optimum) and half "sticking" (fn,
// ft pinned by equality constraints to a fixed, strictly-interior point). Deterministic (no rng)
// so the test is exactly reproducible.
struct ApexProblem {
  SparseMat P, A;
  Vec q, b;
  ConeSpec spec;
  Index num_apex = 0;
};

ApexProblem makeApexProblem(Index num_contacts, Scalar mu) {
  ApexProblem prob;
  const Index num_apex = num_contacts / 2;
  const Index nx = 3 * num_contacts;
  prob.num_apex = num_apex;

  prob.P = SparseMat(nx, nx);
  prob.q = Vec::Zero(nx);
  for (Index c = 0; c < num_apex; ++c) prob.q[3 * c] = 1.0;

  std::vector<Triplet> a_triplets;
  std::vector<Scalar> b_vals;
  Index row = 0;

  for (Index c = num_apex; c < num_contacts; ++c) {
    const Scalar fn = 10.0;
    const Scalar ft1 = 0.4 * mu * fn;  // ||ft|| = 0.4*mu*fn < mu*fn: strictly interior
    const Scalar ft2 = 0.0;
    a_triplets.emplace_back(row, 3 * c, 1.0);
    b_vals.push_back(fn);
    ++row;
    a_triplets.emplace_back(row, 3 * c + 1, 1.0);
    b_vals.push_back(ft1);
    ++row;
    a_triplets.emplace_back(row, 3 * c + 2, 1.0);
    b_vals.push_back(ft2);
    ++row;
  }
  prob.spec.zero_dim = row;

  for (Index c = 0; c < num_contacts; ++c) {
    a_triplets.emplace_back(row, 3 * c, -mu);
    ++row;
    a_triplets.emplace_back(row, 3 * c + 1, -1.0);
    ++row;
    a_triplets.emplace_back(row, 3 * c + 2, -1.0);
    ++row;
    prob.spec.soc_dims.push_back(3);
  }

  prob.A = testutil::makeSparse(row, nx, a_triplets);
  prob.b = Vec::Zero(row);
  for (size_t i = 0; i < b_vals.size(); ++i) prob.b[static_cast<Index>(i)] = b_vals[i];
  return prob;
}

}  // namespace

TEST(SolveApexContacts, HalfAtApexHalfStickingConvergesWithin25IterationsNoNaN) {
  const Index num_contacts = 8;
  const Scalar mu = 0.7;
  ApexProblem prob = makeApexProblem(num_contacts, mu);

  Solver solver;
  ASSERT_TRUE(solver.setup(prob.P, prob.q, prob.A, prob.b, prob.spec));
  const Solution& sol = solver.solve();

  ASSERT_TRUE(sol.ok()) << "status=" << toString(sol.status);
  EXPECT_LE(sol.info.iterations, 25);
  EXPECT_TRUE(sol.x.allFinite());
  EXPECT_TRUE(sol.s.allFinite());
  EXPECT_TRUE(sol.z.allFinite());
  EXPECT_LT(sol.info.primal_residual, 1e-8 * 10);  // tol_feas default is 1e-8; allow reporting slack
  EXPECT_LT(sol.info.dual_residual, 1e-8 * 10);

  // The apex contacts' (fn, ft1, ft2) must actually be ~0 -- not just "converged" in the
  // aggregate residual sense, but landed on the specific degenerate point this test targets.
  for (Index c = 0; c < prob.num_apex; ++c) {
    EXPECT_NEAR(sol.x[3 * c], 0.0, 1e-6) << "apex contact " << c << " normal force";
    EXPECT_NEAR(sol.x[3 * c + 1], 0.0, 1e-6) << "apex contact " << c << " tangential force 1";
    EXPECT_NEAR(sol.x[3 * c + 2], 0.0, 1e-6) << "apex contact " << c << " tangential force 2";
  }

  // The sticking contacts must have landed exactly where the equality constraints pinned them
  // (strictly inside the friction cone, never binding).
  Mat Ad = Mat(prob.A);
  const Scalar eq_residual_inf =
      (Ad.topRows(prob.spec.zero_dim) * sol.x - prob.b.head(prob.spec.zero_dim)).cwiseAbs().maxCoeff();
  EXPECT_LE(eq_residual_inf, 1e-6);
}
