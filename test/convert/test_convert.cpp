#include "conicxx/convert.h"

#include <gtest/gtest.h>

#include "test_helpers.h"

using namespace conicxx;

TEST(Convert, DenseToSparseKeepsAllEntriesIncludingZeros) {
  Mat A(2, 3);
  A << 1.0, 0.0, 3.0, 0.0, 5.0, 0.0;

  testing::internal::CaptureStderr();
  SparseMat S = toSparse(A);
  const std::string warning = testing::internal::GetCapturedStderr();

  EXPECT_FALSE(warning.empty());
  ASSERT_EQ(S.rows(), 2);
  ASSERT_EQ(S.cols(), 3);
  // Every entry explicit, even the exact zeros -- this is what keeps the
  // derived sparsity pattern stable across repeated conversions.
  EXPECT_EQ(S.nonZeros(), 6);
  for (Index i = 0; i < 2; ++i) {
    for (Index j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(S.coeff(i, j), A(i, j));
    }
  }
}

TEST(Convert, DenseToSparsePatternStableAcrossChangingZeros) {
  // Same structural conversion called twice with a value that happens to be
  // exactly zero the second time -- the resulting nnz/pattern must not
  // change, since Solver::updateData() relies on a stable pattern.
  Mat A1(2, 2);
  A1 << 1.0, 2.0, 3.0, 4.0;
  Mat A2(2, 2);
  A2 << 1.0, 0.0, 3.0, 4.0;  // (0,1) entry is now exactly zero

  SparseMat S1 = toSparse(A1, nullptr, false);
  SparseMat S2 = toSparse(A2, nullptr, false);
  EXPECT_EQ(S1.nonZeros(), S2.nonZeros());
}

TEST(Convert, WarnFalseSuppressesWarning) {
  Mat A = Mat::Identity(2, 2);
  testing::internal::CaptureStderr();
  SparseMat S = toSparse(A, "A", false);
  const std::string warning = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(warning.empty());
  EXPECT_EQ(S.nonZeros(), 4);
}

TEST(Convert, UpperTriangularExtractsOnlyUpperTriangle) {
  Mat P(3, 3);
  P << 4.0, 1.0, 2.0,   //
      1.0, 5.0, 3.0,    //
      2.0, 3.0, 6.0;

  testing::internal::CaptureStderr();
  SparseMat S = toSparseUpperTriangular(P);
  testing::internal::GetCapturedStderr();

  EXPECT_EQ(S.nonZeros(), 6);  // n*(n+1)/2 for n=3
  for (Index i = 0; i < 3; ++i) {
    for (Index j = 0; j < 3; ++j) {
      if (i <= j) {
        EXPECT_DOUBLE_EQ(S.coeff(i, j), P(i, j));
      } else {
        EXPECT_DOUBLE_EQ(S.coeff(i, j), 0.0);
      }
    }
  }
}

TEST(Convert, UpperTriangularWarnsOnAsymmetricInput) {
  Mat P(2, 2);
  P << 1.0, 2.0, 0.0, 1.0;  // not symmetric: P(0,1)=2 != P(1,0)=0

  testing::internal::CaptureStderr();
  toSparseUpperTriangular(P, "P");
  const std::string warning = testing::internal::GetCapturedStderr();
  EXPECT_NE(warning.find("not numerically symmetric"), std::string::npos);
}

TEST(Convert, UpperTriangularNoAsymmetryWarningOnSymmetricInput) {
  Mat P(2, 2);
  P << 1.0, 2.0, 2.0, 1.0;

  testing::internal::CaptureStderr();
  toSparseUpperTriangular(P, "P");
  const std::string warning = testing::internal::GetCapturedStderr();
  EXPECT_EQ(warning.find("not numerically symmetric"), std::string::npos);
}

TEST(Convert, RowMajorSparseConvertsToCanonicalColumnMajor) {
  SparseMatRowMajor csr(2, 2);
  std::vector<Eigen::Triplet<Scalar, Index>> triplets = {{0, 0, 1.0}, {0, 1, 2.0}, {1, 1, 3.0}};
  csr.setFromTriplets(triplets.begin(), triplets.end());

  testing::internal::CaptureStderr();
  SparseMat csc = toSparse(csr);
  const std::string warning = testing::internal::GetCapturedStderr();

  EXPECT_FALSE(warning.empty());
  ASSERT_EQ(csc.rows(), 2);
  ASSERT_EQ(csc.cols(), 2);
  EXPECT_DOUBLE_EQ(csc.coeff(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(csc.coeff(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(csc.coeff(1, 1), 3.0);
  EXPECT_DOUBLE_EQ(csc.coeff(1, 0), 0.0);
}
