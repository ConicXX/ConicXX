#include "conicxx/kkt/sparsity_map.h"

#include <gtest/gtest.h>

using namespace conicxx;
using namespace conicxx::detail;

TEST(SparsityMap, BasicSetGet) {
  SparsityMap sm;
  Index s00 = sm.addEntry(0, 0);
  Index s11 = sm.addEntry(1, 1);
  Index s10 = sm.addEntry(1, 0);
  SparseMat K = sm.finalize(2, 2);

  sm.setValue(K, s00, 3.0);
  sm.setValue(K, s11, 4.0);
  sm.setValue(K, s10, 5.0);

  EXPECT_DOUBLE_EQ(K.coeff(0, 0), 3.0);
  EXPECT_DOUBLE_EQ(K.coeff(1, 1), 4.0);
  EXPECT_DOUBLE_EQ(K.coeff(1, 0), 5.0);
  EXPECT_DOUBLE_EQ(sm.getValue(K, s00), 3.0);
}

TEST(SparsityMap, AliasedSlotsShareOffset) {
  SparsityMap sm;
  Index a = sm.addEntry(2, 3);
  Index b = sm.addEntry(2, 3);  // same physical position, different logical slot
  SparseMat K = sm.finalize(5, 5);

  sm.setValue(K, a, 7.0);
  EXPECT_DOUBLE_EQ(sm.getValue(K, b), 7.0);
  sm.setValue(K, b, 9.0);
  EXPECT_DOUBLE_EQ(sm.getValue(K, a), 9.0);
  EXPECT_DOUBLE_EQ(K.coeff(2, 3), 9.0);
  EXPECT_EQ(K.nonZeros(), 1);
}

TEST(SparsityMap, MultiColumnLayout) {
  SparsityMap sm;
  // Scatter entries across several columns/rows out of order.
  Index s1 = sm.addEntry(3, 1);
  Index s2 = sm.addEntry(0, 0);
  Index s3 = sm.addEntry(4, 2);
  Index s4 = sm.addEntry(1, 1);
  SparseMat K = sm.finalize(5, 5);

  sm.setValue(K, s1, 1.0);
  sm.setValue(K, s2, 2.0);
  sm.setValue(K, s3, 3.0);
  sm.setValue(K, s4, 4.0);

  EXPECT_DOUBLE_EQ(K.coeff(3, 1), 1.0);
  EXPECT_DOUBLE_EQ(K.coeff(0, 0), 2.0);
  EXPECT_DOUBLE_EQ(K.coeff(4, 2), 3.0);
  EXPECT_DOUBLE_EQ(K.coeff(1, 1), 4.0);
  EXPECT_EQ(K.nonZeros(), 4);
}

TEST(SparsityMap, ValuesInitializedToZero) {
  SparsityMap sm;
  Index s = sm.addEntry(1, 1);
  SparseMat K = sm.finalize(3, 3);
  EXPECT_DOUBLE_EQ(sm.getValue(K, s), 0.0);
}
