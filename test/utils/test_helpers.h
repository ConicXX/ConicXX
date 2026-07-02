#pragma once

#include <gtest/gtest.h>

#include <vector>

#include "conicxx/cone_spec.h"
#include "conicxx/cones/cone_set.h"
#include "conicxx/solution.h"
#include "conicxx/types.h"

namespace conicxx::testutil {

inline SparseMat makeSparse(Index rows, Index cols, const std::vector<Triplet>& triplets) {
  SparseMat m(rows, cols);
  m.setFromTriplets(triplets.begin(), triplets.end());
  m.makeCompressed();
  return m;
}

inline void expectVecNear(const Vec& a, const Vec& b, Scalar tol) {
  ASSERT_EQ(a.size(), b.size());
  for (Index i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(a[i], b[i], tol) << "at index " << i;
  }
}

/// Verifies the reported solution actually satisfies the problem's KKT
/// optimality conditions directly (stationarity, primal feasibility,
/// complementarity, cone membership) -- catches sign/formulation bugs that
/// a single hard-coded expected-vector comparison would miss.
inline void expectKktOptimal(const Mat& P, const Vec& q, const Mat& A, const Vec& b,
                             const ConeSpec& spec, const Solution& sol, Scalar tol) {
  ASSERT_TRUE(sol.ok()) << "status=" << toString(sol.status);
  const Vec stationarity = P * sol.x + A.transpose() * sol.z + q;
  const Vec feasibility = A * sol.x + sol.s - b;
  EXPECT_LT(stationarity.norm(), tol) << "stationarity residual too large";
  EXPECT_LT(feasibility.norm(), tol) << "primal feasibility residual too large";
  EXPECT_NEAR(sol.s.dot(sol.z), 0.0, tol) << "complementarity s'z not ~0";

  ConeSet cones(spec);
  const auto [min_margin_s, pos_s] = cones.margins(sol.s);
  const auto [min_margin_z, pos_z] = cones.margins(sol.z);
  (void)pos_s;
  (void)pos_z;
  EXPECT_GT(min_margin_s, -tol) << "s not in cone";
  EXPECT_GT(min_margin_z, -tol) << "z not in dual cone";
}

}  // namespace conicxx::testutil
