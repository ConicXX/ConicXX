#pragma once

#include <utility>
#include <vector>

#include "conicxx/cones/cone_base.h"
#include "conicxx/cone_spec.h"

namespace conicxx {

/// Composite cone K = Zero x Nonnegative x SOC x ... x SOC, in that fixed block order, operating
/// on stacked vectors of total dimension totalDim().
///
/// The scaling data of all blocks live in contiguous arrays owned by this class (orthant: w;
/// SOC: eta per block and w aligned with the block's rows), and every operation is one loop over
/// the blocks without virtual calls; when all SOC blocks are 3-dimensional (friction cones) the
/// loop uses fixed-size kernels. The per-cone classes (ZeroCone, NonnegativeCone,
/// SecondOrderCone) implement the same math for a single block. Inputs and outputs of the
/// vector operations must not alias.
class ConeSet {
 public:
  explicit ConeSet(const ConeSpec& spec);

  Index totalDim() const { return total_dim_; }
  Index numBlocks() const { return static_cast<Index>(block_type_.size()); }
  ConeType blockType(Index i) const { return block_type_[static_cast<size_t>(i)]; }
  Index blockOffset(Index i) const { return block_off_[static_cast<size_t>(i)]; }
  Index blockDim(Index i) const { return block_dim_[static_cast<size_t>(i)]; }
  /// Entries of block i in the lower triangle of Hs: dim for Zero/Nonnegative (diagonal),
  /// dim*(dim+1)/2 for SOC (dense).
  Index numHsEntries(Index i) const;
  /// Hs lower-triangle entries of block i in row-major order (see ConeBase::writeHsLowerTriangle).
  void writeHsLowerTriangle(Index i, Scalar* out) const;

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

  /// Aggregate (min_margin, sum of positive margins) over all non-zero-cone blocks; the orthant
  /// counts as one block with margin min_i x_i.
  std::pair<Scalar, Scalar> margins(const Eigen::Ref<const Vec>& x) const;

  /// Minimum of ConeBase::centrality(s, z) over all blocks; +infinity without non-zero cones.
  Scalar minCentrality(const Vec& s, const Vec& z) const;

  void scaledUnitShift(Eigen::Ref<Vec> x, Scalar alpha) const;

  /// Euclidean projection of the orthant and SOC segments onto their cones; the zero-cone
  /// segment is left unchanged.
  void projectOntoCones(Eigen::Ref<Vec> x) const;

  /// Sets the zero-cone segment of s to exactly 0 (needed at initialization only; Newton steps
  /// keep it there).
  void zeroPrimalZeroConeBlocks(Eigen::Ref<Vec> s) const;

  /// Largest step in [0, alpha_max] keeping x + alpha dx in the closed cone.
  Scalar maxStep(const Vec& x, const Vec& dx, Scalar alpha_max) const;

 private:
  /// Calls f(std::integral_constant<int, D>, k, offset, dim) for every SOC block k, with D = 3 if
  /// all SOC blocks are 3-dimensional and D = -1 (runtime dimension) otherwise.
  template <typename F>
  void forEachSoc(F&& f) const;

  Index zero_dim_ = 0, nonneg_dim_ = 0, total_dim_ = 0, degree_ = 0;
  std::vector<ConeType> block_type_;
  std::vector<Index> block_off_, block_dim_;
  std::vector<Index> soc_off_, soc_dim_;  // per SOC block
  bool all_soc3_ = true;
  Vec identity_;
  Vec nonneg_w_;  // orthant scaling, w_i = sqrt(s_i / z_i)
  Vec soc_eta_;   // per SOC block
  Vec soc_w_;     // size totalDim(); the w of each SOC block at the block's rows
};

}  // namespace conicxx
