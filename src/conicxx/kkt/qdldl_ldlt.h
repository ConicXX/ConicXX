#pragma once

#include <Eigen/Core>
#include <vector>

#include "conicxx/types.h"

namespace conicxx::detail {

/// Wraps QDLDL (github.com/osqp/qdldl, the sparse LDL^T backend QOCO/Clarabel use) behind the
/// same analyzePattern()/factorize()/solve()/info()/vectorD() shape as
/// Eigen::SimplicialLDLT<SparseMat, Eigen::Lower>, so KktSystem can dispatch to either backend
/// through identical call sites.
///
/// QDLDL itself performs no fill-reducing ordering and expects the *upper* triangle of the
/// matrix in CSC form (Eigen::SimplicialLDLT<..., Lower> and the rest of KktSystem use the
/// lower triangle). This wrapper computes the ordering once via Eigen::AMDOrdering -- the same
/// algorithm Eigen::SimplicialLDLT uses internally by default -- and builds the permuted upper
/// triangle via SparseSelfAdjointView::twistedBy(), mirroring exactly what
/// Eigen::SimplicialCholesky does internally on every analyzePattern()/factorize() call (see
/// Eigen/src/SparseCholesky/SimplicialCholesky.h); this is not a new cost relative to the
/// existing Eigen backend, just one now made explicit here.
class QdldlLdlt {
 public:
  /// One-time: computes the AMD ordering and the elimination tree/L sparsity pattern from
  /// K_lower's structure. Must be called again if K_lower's sparsity pattern changes.
  void analyzePattern(const SparseMat& K_lower);

  /// Refreshes the permuted-upper mirror of K_lower's values and runs QDLDL_factor. K_lower
  /// must have the same sparsity pattern passed to the last analyzePattern() call.
  void factorize(const SparseMat& K_lower);

  /// Solves K_lower * x = rhs using the current factorization.
  Vec solve(const Vec& rhs) const;

  Eigen::ComputationInfo info() const { return info_; }

  /// D from the LDL^T factorization, in the *permuted* variable order. Only ever used by
  /// callers for aggregate min/max-magnitude pivot inspection, which is order-independent, so
  /// the permutation is deliberately not undone here.
  Vec vectorD() const { return Eigen::Map<const Vec>(D_.data(), n_); }

 private:
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, Index> perm_;
  SparseMat A_upper_;  // permuted, upper-triangular mirror of K_lower; values refreshed every factorize()

  std::vector<Index> etree_, Lnz_, Lp_, Li_, iwork_;
  std::vector<Scalar> Lx_, D_, Dinv_, fwork_;
  std::vector<unsigned char> bwork_;

  Index n_ = 0;
  Eigen::ComputationInfo info_ = Eigen::NumericalIssue;
};

}  // namespace conicxx::detail
