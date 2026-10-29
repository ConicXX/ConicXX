#include <gtest/gtest.h>

#include "conicxx/solver.h"
#include "test_helpers.h"

using namespace conicxx;

namespace {

// Same box-constrained QP as SolveQp.BoxConstraintActiveAtOneBound: min
// 0.5*(x0+1)^2+0.5*(x1-2)^2 s.t. x0>=0, x1>=0; constrained optimum (0,2).
struct ProblemData {
  Mat P_dense, A_dense;
  SparseMat P_sparse, A_sparse;
  Vec q, b;
  ConeSpec spec;
};

ProblemData makeProblem() {
  ProblemData d;
  d.P_dense = Mat::Identity(2, 2);
  d.q.resize(2);
  d.q << 1.0, -2.0;
  d.A_dense = -Mat::Identity(2, 2);
  d.b = Vec::Zero(2);
  d.spec.nonneg_dim = 2;

  d.P_sparse = testutil::makeSparse(2, 2, {{0, 0, 1.0}, {1, 1, 1.0}});
  d.A_sparse = testutil::makeSparse(2, 2, {{0, 0, -1.0}, {1, 1, -1.0}});
  return d;
}

}  // namespace

TEST(SolveDenseInput, BothDenseMatchesSparseReference) {
  ProblemData d = makeProblem();

  Solver sparse_solver;
  ASSERT_TRUE(sparse_solver.setup(d.P_sparse, d.q, d.A_sparse, d.b, d.spec));
  const Solution ref = sparse_solver.solve();
  ASSERT_TRUE(ref.ok());

  testing::internal::CaptureStderr();
  Solver dense_solver;
  ASSERT_TRUE(dense_solver.setup(d.P_dense, d.q, d.A_dense, d.b, d.spec));
  const std::string warning = testing::internal::GetCapturedStderr();
  EXPECT_FALSE(warning.empty());  // both P and A should have warned

  const Solution sol = dense_solver.solve();
  ASSERT_TRUE(sol.ok());
  testutil::expectVecNear(sol.x, ref.x, 1e-6);
}

TEST(SolveDenseInput, MixedDensePAndSparseA) {
  ProblemData d = makeProblem();

  Solver solver1;
  ASSERT_TRUE(solver1.setup(d.P_dense, d.q, d.A_sparse, d.b, d.spec));
  const Solution sol1 = solver1.solve();
  ASSERT_TRUE(sol1.ok());

  Solver solver2;
  ASSERT_TRUE(solver2.setup(d.P_sparse, d.q, d.A_dense, d.b, d.spec));
  const Solution sol2 = solver2.solve();
  ASSERT_TRUE(sol2.ok());

  Vec expected(2);
  expected << 0.0, 2.0;
  testutil::expectVecNear(sol1.x, expected, 1e-6);
  testutil::expectVecNear(sol2.x, expected, 1e-6);
}
