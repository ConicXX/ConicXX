#include "conicxx/kkt/quasidefinite_ldlt.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <random>
#include <vector>

using namespace conicxx;
using namespace conicxx::detail;

namespace {

struct TestKkt {
  Mat K;  // dense symmetric reference
  std::vector<signed char> signs;
};

// [P A'; A -H]: banded P (rows i % 3 == 0 zero if singular_p), `me` equality rows with zero
// diagonal (absent from the sparse pattern), `mi` cone rows with dense SPD 3x3 blocks in H.
TestKkt makeKkt(Index n, Index me, Index mi, bool singular_p, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<Scalar> U(-1.0, 1.0), U01(0.0, 1.0);
  const Index N = n + me + mi;
  TestKkt t;
  t.K = Mat::Zero(N, N);
  for (Index i = 0; i < n; ++i) {
    if (singular_p && i % 3 == 0) continue;
    t.K(i, i) = 1.0 + U01(rng);
    if (i + 1 < n && (i + 1) % 3 != 0) t.K(i, i + 1) = t.K(i + 1, i) = 0.3 * U(rng);
  }
  for (Index r = n; r < N; ++r) {
    bool any = false;
    for (Index c = 0; c < n; ++c) {
      if (U01(rng) < 0.3) {
        t.K(r, c) = t.K(c, r) = U(rng);
        any = true;
      }
    }
    if (!any) t.K(r, r % n) = t.K(r % n, r) = 1.0;
  }
  for (Index r = n + me; r < N; r += 3) {
    const Index d = std::min<Index>(3, N - r);
    Mat B(d, d);
    for (Index i = 0; i < d; ++i)
      for (Index j = 0; j < d; ++j) B(i, j) = U(rng);
    t.K.block(r, r, d, d) = -(B.transpose() * B + 1e-3 * Mat::Identity(d, d));
  }
  t.signs.assign(static_cast<size_t>(N), -1);
  for (Index i = 0; i < n; ++i) t.signs[static_cast<size_t>(i)] = 1;
  return t;
}

SparseMat lowerOf(const Mat& K) {
  SparseMat S = K.sparseView();
  SparseMat L = S.triangularView<Eigen::Lower>();
  L.makeCompressed();
  return L;
}

Vec randomVec(Index n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<Scalar> U(-1.0, 1.0);
  Vec v(n);
  for (Index i = 0; i < n; ++i) v[i] = U(rng);
  return v;
}

Scalar relResidual(const Mat& K, const Vec& x, const Vec& b) {
  return (b - K * x).lpNorm<Eigen::Infinity>() / b.lpNorm<Eigen::Infinity>();
}

}  // namespace

TEST(QuasiDefiniteLdlt, ExactOnQuasidefiniteMatrices) {
  // P > 0 and only cone rows (H > 0): every pivot has its expected sign without any
  // regularization, so the factorization is exact.
  for (unsigned seed = 0; seed < 10; ++seed) {
    const TestKkt t = makeKkt(20, 0, 15, false, seed);
    const SparseMat K = lowerOf(t.K);
    QuasiDefiniteLdlt ldl;
    ldl.params.refine_max_iter = 0;
    ASSERT_TRUE(ldl.analyzePattern(K, t.signs));
    ASSERT_TRUE(ldl.factorize(K));
    EXPECT_EQ(ldl.numDynamicRegularized(), 0);
    EXPECT_EQ(ldl.numPositivePivots(), 20);
    const Vec b = randomVec(K.rows(), seed);
    Vec x;
    ldl.solve(b, x);
    const Vec xd = t.K.partialPivLu().solve(b);
    EXPECT_LT((x - xd).lpNorm<Eigen::Infinity>(), 1e-9 * (1.0 + xd.lpNorm<Eigen::Infinity>()))
        << "seed " << seed;
  }
}

TEST(QuasiDefiniteLdlt, StaticRegularizationIsRemovedByRefinement) {
  // Singular P and equality rows with structurally absent diagonal: the static regularization
  // makes the factorization possible, refinement against the exact matrix removes it.
  int compared = 0;
  for (unsigned seed = 0; seed < 10; ++seed) {
    const TestKkt t = makeKkt(30, 6, 18, true, seed);
    if (Eigen::FullPivLU<Mat>(t.K).rank() < t.K.rows()) continue;
    const SparseMat K = lowerOf(t.K);
    QuasiDefiniteLdlt ldl;
    ASSERT_TRUE(ldl.analyzePattern(K, t.signs));
    ldl.setStaticRegularization(Vec::Constant(K.rows(), 1e-6));
    ASSERT_TRUE(ldl.factorize(K));
    EXPECT_EQ(ldl.numPositivePivots(), 30);
    const Vec b = randomVec(K.rows(), 100 + seed);
    Vec x;
    ldl.params.refine_max_iter = 0;
    ldl.solve(b, x);
    const Scalar unrefined = relResidual(t.K, x, b);
    ldl.params.refine_max_iter = 10;
    const Scalar res = ldl.solve(b, x);
    EXPECT_LT(relResidual(t.K, x, b), 1e-12) << "seed " << seed;
    EXPECT_LE(relResidual(t.K, x, b), unrefined);
    EXPECT_NEAR(res, (b - t.K * x).lpNorm<Eigen::Infinity>(), 1e-12);
    EXPECT_GT(ldl.lastRefineSteps(), 0);
    ++compared;
  }
  EXPECT_GE(compared, 5);
}

TEST(QuasiDefiniteLdlt, DynamicRegularizationKeepsSingularMatrixFactorizable) {
  // Two identical equality rows and no static regularization: the second row's pivot vanishes.
  TestKkt t = makeKkt(10, 3, 6, false, 7);
  const Index n = 10;
  t.K.block(n + 1, 0, 1, n) = t.K.block(n, 0, 1, n);
  t.K.block(0, n + 1, n, 1) = t.K.block(0, n, n, 1);
  const SparseMat K = lowerOf(t.K);
  QuasiDefiniteLdlt ldl;
  ASSERT_TRUE(ldl.analyzePattern(K, t.signs));
  ASSERT_TRUE(ldl.factorize(K));
  EXPECT_GT(ldl.numDynamicRegularized(), 0);
  EXPECT_EQ(ldl.numPositivePivots(), n);

  ldl.params.dynamic_enable = false;
  EXPECT_FALSE(ldl.factorize(K));
}

TEST(QuasiDefiniteLdlt, IgnoresUpperTriangleAndRefactorizes) {
  const TestKkt t = makeKkt(40, 5, 24, false, 3);
  SparseMat full = t.K.sparseView();
  full.makeCompressed();
  const SparseMat lower = lowerOf(t.K);
  QuasiDefiniteLdlt a, b;
  ASSERT_TRUE(a.analyzePattern(full, t.signs));
  ASSERT_TRUE(b.analyzePattern(lower, t.signs));
  a.setStaticRegularization(Vec::Constant(full.rows(), 1e-8));
  b.setStaticRegularization(Vec::Constant(full.rows(), 1e-8));
  ASSERT_TRUE(a.factorize(full));
  ASSERT_TRUE(b.factorize(lower));
  const Vec rhs = randomVec(full.rows(), 5);
  Vec xa, xb;
  a.solve(rhs, xa);
  b.solve(rhs, xb);
  EXPECT_LT((xa - xb).lpNorm<Eigen::Infinity>(), 1e-10 * (1.0 + xa.lpNorm<Eigen::Infinity>()));

  // New values on the same pattern, as in every IPM iteration.
  SparseMat scaled = lower;
  for (Index p = 0; p < scaled.nonZeros(); ++p) scaled.valuePtr()[p] *= 1.5;
  ASSERT_TRUE(b.factorize(scaled));
  b.solve(rhs, xb);
  EXPECT_LT(relResidual(1.5 * t.K, xb, rhs), 1e-12);
}

TEST(QuasiDefiniteLdlt, ReportsNonFiniteDataAndPatternMismatch) {
  TestKkt t = makeKkt(8, 2, 6, false, 11);
  SparseMat K = lowerOf(t.K);
  QuasiDefiniteLdlt ldl;
  ASSERT_TRUE(ldl.analyzePattern(K, t.signs));
  ASSERT_TRUE(ldl.factorize(K));
  K.valuePtr()[0] = std::numeric_limits<Scalar>::quiet_NaN();
  EXPECT_FALSE(ldl.factorize(K));

  SparseMat other = lowerOf(makeKkt(8, 2, 6, false, 12).K);
  if (other.nonZeros() != K.nonZeros()) {
    EXPECT_FALSE(ldl.factorize(other));
  }
}
