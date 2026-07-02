#pragma once

#include <memory>
#include <vector>

#include "conicxx/cone_spec.h"
#include "conicxx/cones/cone_base.h"

namespace conicxx {

/// Composite cone K = Zero x Nonnegative x SOC x ... x SOC, in that fixed
/// block order. Provides aggregate operations that loop over blocks and
/// dispatch to each ConeBase, operating on stacked vectors of total
/// dimension `totalDim()`.
class ConeSet {
 public:
  explicit ConeSet(const ConeSpec& spec);

  Index totalDim() const { return total_dim_; }
  Index numBlocks() const { return static_cast<Index>(cones_.size()); }
  const ConeBase& block(Index i) const { return *cones_[static_cast<size_t>(i)]; }
  Index blockOffset(Index i) const { return offsets_[static_cast<size_t>(i)]; }

  /// Sum of per-block degrees (0 for Zero, dim for Nonnegative, 1 per SOC block).
  Index degree() const { return degree_; }

  const Vec& identityElement() const { return identity_; }

  void product(const Vec& u, const Vec& v, Eigen::Ref<Vec> out) const;
  void inverseProduct(const Vec& u, const Vec& w, Eigen::Ref<Vec> out) const;

  void updateScaling(const Vec& s, const Vec& z);

  void applyW(const Vec& x, Eigen::Ref<Vec> out) const;
  void applyWInv(const Vec& x, Eigen::Ref<Vec> out) const;

  /// Hs * x where Hs is the block-diagonal aggregate scaling matrix.
  void mulHs(const Vec& x, Eigen::Ref<Vec> out) const;

  /// Per-block dense Hs matrices, in block order, for KKT assembly.
  const std::vector<Mat>& scalingBlocks() const { return scaling_blocks_; }

  /// Aggregate (min_margin, sum of positive margins) over all blocks.
  std::pair<Scalar, Scalar> margins(const Vec& x) const;

  void scaledUnitShift(Eigen::Ref<Vec> x, Scalar alpha) const;

  /// Force the primal-slack segments belonging to Zero-cone blocks to
  /// exactly 0. Needed only at initialization: the Newton step keeps those
  /// segments invariant at whatever they start at (Delta s is provably 0
  /// there given the Zero cone's degenerate Jordan algebra), but the
  /// initial KKT-solve-based point does not itself produce exact zeros.
  void zeroPrimalZeroConeBlocks(Eigen::Ref<Vec> s) const;

  /// Combined per-cone-block-limited step length, minimized over blocks.
  Scalar maxStep(const Vec& x, const Vec& dx, Scalar alpha_max) const;

 private:
  std::vector<std::unique_ptr<ConeBase>> cones_;
  std::vector<Index> offsets_;
  std::vector<Mat> scaling_blocks_;
  Index total_dim_ = 0;
  Index degree_ = 0;
  Vec identity_;
};

}  // namespace conicxx
