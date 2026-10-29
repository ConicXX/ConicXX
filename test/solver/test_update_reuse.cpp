#include "conicxx/solver.h"

#include <gtest/gtest.h>

#include "test_helpers.h"

using namespace conicxx;

namespace {

// A little "timestep" of a box-constrained QP: min 0.5||x-c||^2 s.t. x>=0,
// parametrized by the target point c. Closed form: x_i = max(c_i, 0).
struct Problem {
  SparseMat P, A;
  Vec q, b;
  ConeSpec spec;
};

Problem makeProblem(const Vec& c) {
  Problem p;
  const Index n = static_cast<Index>(c.size());
  p.P = testutil::makeSparse(n, n, {{0, 0, 1.0}, {1, 1, 1.0}});
  p.q = -c;
  p.A = testutil::makeSparse(n, n, {{0, 0, -1.0}, {1, 1, -1.0}});
  p.b = Vec::Zero(n);
  p.spec.nonneg_dim = n;
  return p;
}

}  // namespace

TEST(UpdateReuse, RepeatedUpdateDataMatchesFreshSetupEachTime) {
  std::vector<Vec> targets;
  {
    Vec c(2);
    c << 1.0, 2.0;
    targets.push_back(c);
  }
  {
    Vec c(2);
    c << -1.0, 3.0;
    targets.push_back(c);
  }
  {
    Vec c(2);
    c << 0.5, -0.5;
    targets.push_back(c);
  }

  Problem p0 = makeProblem(targets[0]);
  Solver reused;
  ASSERT_TRUE(reused.setup(p0.P, p0.q, p0.A, p0.b, p0.spec));

  for (size_t t = 0; t < targets.size(); ++t) {
    Problem p = makeProblem(targets[t]);

    if (t > 0) {
      ASSERT_TRUE(reused.updateData(nullptr, &p.q, nullptr, nullptr));
    }
    const Solution& sol_reused = reused.solve();
    ASSERT_TRUE(sol_reused.ok());

    Solver fresh;
    ASSERT_TRUE(fresh.setup(p.P, p.q, p.A, p.b, p.spec));
    const Solution& sol_fresh = fresh.solve();
    ASSERT_TRUE(sol_fresh.ok());

    testutil::expectVecNear(sol_reused.x, sol_fresh.x, 1e-6);

    Vec expected = targets[t].cwiseMax(0.0);
    testutil::expectVecNear(sol_reused.x, expected, 1e-6);
  }
}

TEST(UpdateReuse, PatternMismatchIsRejectedAndRequiresFullSetup) {
  Problem p0 = makeProblem((Vec(2) << 1.0, 1.0).finished());
  Solver solver;
  ASSERT_TRUE(solver.setup(p0.P, p0.q, p0.A, p0.b, p0.spec));

  // A structurally different A (extra nonzero) must be rejected by
  // updateData -- the caller is expected to fall back to setup().
  SparseMat A_bad =
      testutil::makeSparse(2, 2, {{0, 0, -1.0}, {0, 1, 0.3}, {1, 1, -1.0}});
  EXPECT_FALSE(solver.updateData(nullptr, nullptr, &A_bad, nullptr));

  ASSERT_TRUE(solver.setup(p0.P, p0.q, A_bad, p0.b, p0.spec));
  const Solution& sol = solver.solve();
  EXPECT_TRUE(sol.ok());
}
