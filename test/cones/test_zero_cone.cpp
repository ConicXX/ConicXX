#include "conicxx/cones/zero_cone.h"

#include <gtest/gtest.h>

using namespace conicxx;

TEST(ZeroCone, BasicProperties) {
  ZeroCone cone(4);
  EXPECT_EQ(cone.dim(), 4);
  EXPECT_EQ(cone.type(), ConeType::Zero);
  EXPECT_EQ(cone.degree(), 0);
  EXPECT_TRUE(cone.identityElement().isZero());
}

TEST(ZeroCone, ScalingBlockIsZero) {
  ZeroCone cone(3);
  Vec s = Vec::Random(3), z = Vec::Random(3);
  cone.updateScaling(s, z);
  EXPECT_TRUE(cone.scalingBlock().isZero());
}

TEST(ZeroCone, ApplyWIsZero) {
  ZeroCone cone(3);
  Vec x = Vec::Random(3), out(3);
  cone.applyW(x, out);
  EXPECT_TRUE(out.isZero());
  cone.applyWInv(x, out);
  EXPECT_TRUE(out.isZero());
}

TEST(ZeroCone, ProductIsZero) {
  ZeroCone cone(3);
  Vec u = Vec::Random(3), v = Vec::Random(3), out(3);
  cone.product(u, v, out);
  EXPECT_TRUE(out.isZero());
  cone.inverseProduct(u, v, out);
  EXPECT_TRUE(out.isZero());
}

TEST(ZeroCone, MarginIsInfinity) {
  ZeroCone cone(2);
  Vec x = Vec::Random(2);
  EXPECT_TRUE(std::isinf(cone.margin(x)));
}

TEST(ZeroCone, ShiftIsNoOp) {
  ZeroCone cone(3);
  Vec x = Vec::Random(3);
  Vec x0 = x;
  cone.scaledUnitShift(x, 5.0);
  EXPECT_EQ(x, x0);
}

TEST(ZeroCone, MaxStepUnconstrained) {
  ZeroCone cone(3);
  Vec x = Vec::Random(3), dx = Vec::Random(3);
  EXPECT_EQ(cone.maxStep(x, dx, 0.73), 0.73);
}
