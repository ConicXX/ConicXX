// T2.4 (CONICXX_AGENT_TASKS.md): RegularizedLdlt must match QdldlLdlt when no pivot needs
// correction, and must solve exactly the additively-shifted system when one does. Only built
// when CONICXX_HAVE_QDLDL is defined (both backends require QDLDL).
#ifdef CONICXX_HAVE_QDLDL

#include "conicxx/kkt/regularized_ldlt.h"

#include <gtest/gtest.h>

#include <Eigen/Cholesky>
#include <vector>

#include "conicxx/kkt/qdldl_ldlt.h"
#include "test_helpers.h"

using namespace conicxx;
using namespace conicxx::detail;

TEST(RegularizedLdlt, MatchesQdldlLdltWhenNoPivotNeedsCorrection) {
  // A well-conditioned quasi-definite matrix (P block positive definite, -Hs block negative
  // definite, both with margin well above dynamic_reg_eps) that no backend should ever need to
  // touch a pivot for -- RegularizedLdlt's inline correction and QdldlLdlt's plain factorization
  // should then agree with each other (and with a dense reference) to near machine precision.
  const Index n = 5;
  SparseMat K_lower = testutil::makeSparse(
      n, n,
      {{0, 0, 4.0},
       {1, 1, 3.0},
       {2, 0, 0.5}, {2, 2, -2.0},
       {3, 1, 0.3}, {3, 2, 0.2}, {3, 3, -3.0},
       {4, 0, -0.4}, {4, 3, 0.1}, {4, 4, -1.5}});
  std::vector<unsigned char> is_zero_row(static_cast<size_t>(n), 0);  // no zero-cone rows here
  const Index nx = 2;  // rows/cols [0,2) are the "P block", [2,5) the "-Hs block"
  const Scalar eps = 1e-13, delta = 2e-7;

  QdldlLdlt qdldl;
  qdldl.analyzePattern(K_lower);
  qdldl.factorize(K_lower);
  ASSERT_EQ(qdldl.info(), Eigen::Success);

  RegularizedLdlt reg;
  reg.analyzePattern(K_lower, is_zero_row, nx, eps, delta);
  reg.factorize(K_lower);
  ASSERT_EQ(reg.info(), Eigen::Success);
  EXPECT_EQ(reg.numRegularizedPivots(), 0);

  Vec rhs(n);
  rhs << 1.0, -2.0, 0.5, 3.0, -1.0;
  const Vec x_qdldl = qdldl.solve(rhs);
  const Vec x_reg = reg.solve(rhs);
  testutil::expectVecNear(x_reg, x_qdldl, 1e-12);

  // Both should also match a plain dense reference solve.
  Mat Kd = Mat::Zero(n, n);
  for (Index c = 0; c < K_lower.outerSize(); ++c) {
    for (SparseMat::InnerIterator it(K_lower, c); it; ++it) {
      Kd(it.row(), it.col()) = it.value();
      Kd(it.col(), it.row()) = it.value();
    }
  }
  const Vec x_dense = Kd.ldlt().solve(rhs);
  testutil::expectVecNear(x_reg, x_dense, 1e-10);
}

TEST(RegularizedLdlt, ForcedCorrectionSolvesAdditivelyShiftedSystemExactly) {
  // A purely diagonal K (no off-diagonal coupling at all) so the factorization is
  // order-independent and hand-verifiable regardless of which permutation AMD picks: D at
  // whichever position a given original row lands is exactly that row's own diagonal entry, no
  // Schur-complement contributions from anywhere. Row 0 (the "P block", nx=1) has d0 = 1e-15 --
  // nonzero but far below dynamic_reg_eps, so it needs correction; every other row is a
  // comfortably-conditioned Hs-block entry. n = 30 keeps the single corrected pivot's fraction
  // (1/30) below kMaxRegularizedFraction (0.05, see regularized_ldlt.cpp) so info() still
  // reports Success instead of falling back to KktSystem's outer retry loop.
  const Index n = 30;
  const Scalar d0 = 1e-15;
  std::vector<Triplet> triplets;
  triplets.emplace_back(0, 0, d0);
  for (Index i = 1; i < n; ++i) triplets.emplace_back(i, i, Scalar(-(1 + i)));
  SparseMat K_lower = testutil::makeSparse(n, n, triplets);
  std::vector<unsigned char> is_zero_row(static_cast<size_t>(n), 0);
  const Index nx = 1;
  const Scalar eps = 1e-13, delta = 2e-7;

  RegularizedLdlt reg;
  reg.analyzePattern(K_lower, is_zero_row, nx, eps, delta);
  reg.factorize(K_lower);
  ASSERT_EQ(reg.info(), Eigen::Success);
  ASSERT_EQ(reg.numRegularizedPivots(), 1);

  // K with `delta` added to (0,0) alone -- the additive correction, not a hard replacement (which
  // would instead give (0,0) = delta, discarding d0's 1e-15 entirely; d0 and delta differ by ~8
  // orders of magnitude, so the two are easily distinguishable at solve precision).
  Vec rhs = Vec::LinSpaced(n, 0.3, -1.7);
  Vec x_shifted_expected(n);
  x_shifted_expected[0] = rhs[0] / (d0 + delta);
  for (Index i = 1; i < n; ++i) x_shifted_expected[i] = rhs[i] / Scalar(-(1 + i));

  const Vec x_reg = reg.solve(rhs);
  testutil::expectVecNear(x_reg, x_shifted_expected, 1e-9);

  Vec x_replaced_expected = x_shifted_expected;
  x_replaced_expected[0] = rhs[0] / delta;  // hard-replacement hypothesis, for contrast
  EXPECT_GT((x_reg - x_replaced_expected).norm(), 1e-6);
}

#endif  // CONICXX_HAVE_QDLDL
