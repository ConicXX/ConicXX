#include "conicxx/cones/second_order_cone.h"

#include <gtest/gtest.h>

#include "test_helpers.h"

using namespace conicxx;

TEST(SecondOrderCone, BasicProperties) {
  SecondOrderCone cone(3);
  EXPECT_EQ(cone.dim(), 3);
  EXPECT_EQ(cone.type(), ConeType::SecondOrder);
  EXPECT_EQ(cone.degree(), 1);
  Vec e(3);
  e << 1, 0, 0;
  testutil::expectVecNear(cone.identityElement(), e, 1e-14);
}

TEST(SecondOrderCone, ProductAndInverseProductRoundtrip) {
  SecondOrderCone cone(4);
  Vec u(4), v(4);
  u << 3.0, 0.5, -0.2, 0.1;   // interior-ish, not required to be in cone for product test
  v << 2.0, 0.3, 0.4, -0.1;
  Vec w(4);
  cone.product(u, v, w);

  Vec expected(4);
  expected[0] = u.dot(v);
  expected.tail(3) = u[0] * v.tail(3) + v[0] * u.tail(3);
  testutil::expectVecNear(w, expected, 1e-13);

  Vec back(4);
  cone.inverseProduct(u, w, back);  // u \ (u o v) == v, provided u is invertible (u0^2!=||u1||^2)
  testutil::expectVecNear(back, v, 1e-10);
}

TEST(SecondOrderCone, IdentityIsProductNeutral) {
  SecondOrderCone cone(4);
  Vec u(4);
  u << 2.0, 0.3, -0.1, 0.4;
  Vec out(4);
  cone.product(u, cone.identityElement(), out);
  testutil::expectVecNear(out, u, 1e-13);
}

TEST(SecondOrderCone, NTScalingIdentities) {
  SecondOrderCone cone(3);
  Vec s(3), z(3);
  s << 3.0, 1.0, 1.0;   // interior: 3 > sqrt(2)
  z << 2.0, 0.5, 0.5;   // interior: 2 > sqrt(0.5)
  cone.updateScaling(s, z);

  // Defining property of NT scaling: W^2 z == s (equivalently Hs*z == s).
  Vec Hsz(3);
  cone.mulHs(z, Hsz);
  testutil::expectVecNear(Hsz, s, 1e-10);

  // lambda = W^-1 s == W z
  Vec lambda_from_s(3), lambda_from_z(3);
  cone.applyWInv(s, lambda_from_s);
  cone.applyW(z, lambda_from_z);
  testutil::expectVecNear(lambda_from_s, lambda_from_z, 1e-10);

  // Hs == W^T W == W*W is symmetric: x'(Hs y) == y'(Hs x) for any x, y.
  Vec x(3), y(3);
  x << 0.4, -0.2, 0.7;
  y << -0.1, 0.9, 0.3;
  Vec Hsx(3), Hsy(3);
  cone.mulHs(x, Hsx);
  cone.mulHs(y, Hsy);
  EXPECT_NEAR(x.dot(Hsy), y.dot(Hsx), 1e-12);

  // applyW fast-path formula agrees with the cached dense scaling matrix
  // constructed via the same NT-scaling data (self-consistency check).
  Vec Wx_fast(3);
  cone.applyW(x, Wx_fast);
  Vec WWx(3);
  cone.applyW(x, WWx);
  Vec WWx2(3);
  cone.applyW(WWx, WWx2);
  testutil::expectVecNear(WWx2, Hsx, 1e-10);
}

TEST(SecondOrderCone, WriteHsLowerTriangleMatchesMulHs) {
  SecondOrderCone cone(3);
  Vec s(3), z(3);
  s << 3.0, 1.0, 1.0;
  z << 2.0, 0.5, 0.5;
  cone.updateScaling(s, z);

  ASSERT_EQ(cone.numHsEntries(), 6);  // dim*(dim+1)/2
  Vec entries(cone.numHsEntries());
  cone.writeHsLowerTriangle(entries);

  // Reconstruct the dense symmetric Hs from the row-major lower-triangle entries and check it
  // reproduces the same matvec as the structured mulHs() -- the "structured scaling is
  // numerically identical to dense" cross-check from CONICXX_AGENT_TASKS.md Phase 1.
  Mat Hs = Mat::Zero(3, 3);
  Index idx = 0;
  for (Index a = 0; a < 3; ++a) {
    for (Index b = 0; b <= a; ++b) {
      Hs(a, b) = entries[idx];
      Hs(b, a) = entries[idx];
      ++idx;
    }
  }

  Vec x(3);
  x << 0.4, -0.2, 0.7;
  Vec Hsx_expected(3), Hsx_from_entries(3);
  cone.mulHs(x, Hsx_expected);
  Hsx_from_entries.noalias() = Hs * x;
  testutil::expectVecNear(Hsx_from_entries, Hsx_expected, 1e-12);
}

TEST(SecondOrderCone, MarginAndShift) {
  SecondOrderCone cone(3);
  Vec x(3);
  x << 3.0, 1.0, 1.0;
  EXPECT_NEAR(cone.margin(x), 3.0 - std::sqrt(2.0), 1e-14);
  cone.scaledUnitShift(x, 2.0);
  Vec expected(3);
  expected << 5.0, 1.0, 1.0;
  testutil::expectVecNear(x, expected, 1e-14);
}

TEST(SecondOrderCone, MaxStepExactBoundaryHit) {
  SecondOrderCone cone(3);
  Vec x(3);
  x << 1.0, 0.0, 0.0;
  Vec dx(3);
  dx << -1.0, 0.0, 0.0;
  EXPECT_NEAR(cone.maxStep(x, dx, 10.0), 1.0, 1e-12);
}

TEST(SecondOrderCone, MaxStepUnconstrainedWhenMovingAway) {
  SecondOrderCone cone(3);
  Vec x(3);
  x << 1.0, 0.0, 0.0;
  Vec dx(3);
  dx << 1.0, 1.0, 0.0;
  EXPECT_NEAR(cone.maxStep(x, dx, 10.0), 10.0, 1e-12);
}

TEST(SecondOrderCone, MaxStepTangentialPush) {
  SecondOrderCone cone(3);
  Vec x(3);
  x << 1.0, 0.0, 0.0;
  Vec dx(3);
  dx << 0.0, 1.0, 0.0;
  // x + alpha*dx = (1, alpha, 0); leaves cone once alpha > 1.
  EXPECT_NEAR(cone.maxStep(x, dx, 10.0), 1.0, 1e-12);
}
