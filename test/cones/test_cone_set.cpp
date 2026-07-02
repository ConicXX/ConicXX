#include "conicxx/cones/cone_set.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include "test_helpers.h"

using namespace conicxx;

TEST(ConeSet, LayoutAndDegree) {
  ConeSpec spec;
  spec.zero_dim = 2;
  spec.nonneg_dim = 3;
  spec.soc_dims = {3, 4};
  ConeSet cones(spec);

  EXPECT_EQ(cones.numBlocks(), 4);
  EXPECT_EQ(cones.totalDim(), 2 + 3 + 3 + 4);
  EXPECT_EQ(cones.blockOffset(0), 0);
  EXPECT_EQ(cones.blockOffset(1), 2);
  EXPECT_EQ(cones.blockOffset(2), 5);
  EXPECT_EQ(cones.blockOffset(3), 8);

  // degree = 0 (zero) + 3 (nonneg) + 1 + 1 (two SOC blocks)
  EXPECT_EQ(cones.degree(), 5);
}

TEST(ConeSet, IdentityElementLayout) {
  ConeSpec spec;
  spec.zero_dim = 1;
  spec.nonneg_dim = 2;
  spec.soc_dims = {3};
  ConeSet cones(spec);

  Vec expected(6);
  expected << 0.0, 1.0, 1.0, 1.0, 0.0, 0.0;
  testutil::expectVecNear(cones.identityElement(), expected, 1e-14);
}

TEST(ConeSet, MulHsMatchesPerBlockScalingBlocks) {
  ConeSpec spec;
  spec.zero_dim = 1;
  spec.nonneg_dim = 2;
  spec.soc_dims = {3};
  ConeSet cones(spec);

  Vec s(6), z(6);
  s << 0.0, 1.0, 4.0, 3.0, 1.0, 1.0;
  z << 0.0, 2.0, 1.0, 2.0, 0.5, 0.5;
  cones.updateScaling(s, z);

  Vec out(6);
  cones.mulHs(z, out);
  // Hs*z should reproduce s exactly on the nonneg/SOC blocks (NT-scaling
  // defining property), and be identically zero on the zero-cone block.
  EXPECT_NEAR(out[0], 0.0, 1e-14);
  testutil::expectVecNear(out.tail(5), s.tail(5), 1e-10);
}

TEST(ConeSet, MarginsIgnoreZeroCone) {
  ConeSpec spec;
  spec.zero_dim = 2;
  spec.nonneg_dim = 2;
  ConeSet cones(spec);

  Vec x(4);
  x << 100.0, -100.0, -1.0, 3.0;  // zero-cone entries must not affect the aggregate
  auto [min_margin, pos_sum] = cones.margins(x);
  // NonnegativeCone::margin() is min-over-block (not per-component), so the
  // single nonneg block (entries -1, 3) contributes exactly one margin
  // value, -1, and its positive part is 0.
  EXPECT_DOUBLE_EQ(min_margin, -1.0);
  EXPECT_DOUBLE_EQ(pos_sum, 0.0);
}

TEST(ConeSet, MarginsAggregatePositivePartAcrossBlocks) {
  ConeSpec spec;
  spec.nonneg_dim = 2;
  spec.soc_dims = {3, 3};
  ConeSet cones(spec);

  Vec x(8);
  x << 1.0, 2.0,          // nonneg block: margin = 1
      3.0, 1.0, 1.0,      // SOC block: margin = 3 - sqrt(2)
      0.5, 0.4, 0.4;      // SOC block: margin = 0.5 - sqrt(0.32) < 0
  auto [min_margin, pos_sum] = cones.margins(x);
  const Scalar m0 = 1.0;
  const Scalar m1 = 3.0 - std::sqrt(2.0);
  const Scalar m2 = 0.5 - std::sqrt(0.32);
  EXPECT_NEAR(min_margin, std::min({m0, m1, m2}), 1e-12);
  EXPECT_NEAR(pos_sum, std::max(m0, 0.0) + std::max(m1, 0.0) + std::max(m2, 0.0), 1e-12);
}
