#include "conicxx/cones/nonnegative_cone.h"

#include <gtest/gtest.h>

#include "test_helpers.h"

using namespace conicxx;

TEST(NonnegativeCone, BasicProperties) {
  NonnegativeCone cone(5);
  EXPECT_EQ(cone.dim(), 5);
  EXPECT_EQ(cone.type(), ConeType::Nonnegative);
  EXPECT_EQ(cone.degree(), 5);
  EXPECT_TRUE(cone.identityElement().isOnes());
}

TEST(NonnegativeCone, ProductAndInverseProductRoundtrip) {
  NonnegativeCone cone(4);
  Vec u(4);
  u << 1.0, 2.0, 3.0, 4.0;
  Vec v(4);
  v << 5.0, 6.0, 7.0, 8.0;
  Vec w(4);
  cone.product(u, v, w);
  Vec expected = (u.array() * v.array()).matrix();
  testutil::expectVecNear(w, expected, 1e-14);

  Vec back(4);
  cone.inverseProduct(u, w, back);  // u \ (u o v) == v
  testutil::expectVecNear(back, v, 1e-12);
}

TEST(NonnegativeCone, NTScalingSatisfiesW2zEqualsS) {
  NonnegativeCone cone(4);
  Vec s(4), z(4);
  s << 1.0, 4.0, 0.5, 9.0;
  z << 2.0, 1.0, 2.0, 3.0;
  cone.updateScaling(s, z);

  Vec Wz(4);
  cone.applyW(z, Wz);
  Vec HsZ(4);
  HsZ.array() = cone.scalingBlock().diagonal().array() * z.array();
  testutil::expectVecNear(HsZ, s, 1e-12);  // Hs*z == s identically

  Vec lambda_from_s(4), lambda_from_z(4);
  cone.applyWInv(s, lambda_from_s);
  cone.applyW(z, lambda_from_z);
  testutil::expectVecNear(lambda_from_s, lambda_from_z, 1e-12);  // W^-1 s == W z
}

TEST(NonnegativeCone, MarginAndShift) {
  NonnegativeCone cone(3);
  Vec x(3);
  x << -1.0, 2.0, 0.5;
  EXPECT_DOUBLE_EQ(cone.margin(x), -1.0);
  cone.scaledUnitShift(x, 2.0);
  Vec expected(3);
  expected << 1.0, 4.0, 2.5;
  testutil::expectVecNear(x, expected, 1e-14);
}

TEST(NonnegativeCone, MaxStepFractionToBoundary) {
  NonnegativeCone cone(3);
  Vec x(3);
  x << 1.0, 2.0, 3.0;
  Vec dx(3);
  dx << -2.0, -1.0, 5.0;  // binding: x0/(-dx0)=0.5, x1/(-dx1)=2.0
  Scalar alpha = cone.maxStep(x, dx, 10.0);
  EXPECT_NEAR(alpha, 0.5, 1e-14);
}
