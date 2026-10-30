#include "conicxx/kkt/kkt_system.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace conicxx::detail {

namespace {

bool samePattern(const SparseMat& mat, const std::vector<Index>& outer,
                 const std::vector<Index>& inner) {
  if (!mat.isCompressed()) return false;
  if (mat.outerSize() + 1 != static_cast<Index>(outer.size())) return false;
  if (mat.nonZeros() != static_cast<Index>(inner.size())) return false;
  return std::equal(outer.begin(), outer.end(), mat.outerIndexPtr()) &&
         std::equal(inner.begin(), inner.end(), mat.innerIndexPtr());
}

// An equality row of a Newton system counts as unsatisfied above this relative residual.
constexpr Scalar kEqualityResidualTol = 1e-8;

}  // namespace

bool KktSystem::setup(const SparseMat& P_upper, const SparseMat& A, const ConeSet& cones,
                      const Settings& settings) {
  n_ = static_cast<Index>(P_upper.rows());
  m_ = cones.totalDim();
  const Index dim = n_ + m_;
  factorized_ = false;
  last_refinement_residual_ = 0;
  stats_ = Stats{};
  compliance_.resize(0);
  if (!P_upper.isCompressed() || !A.isCompressed()) return false;

  std::vector<signed char> signs(static_cast<size_t>(dim), -1);
  std::vector<Index> zero_rows;
  reg_base_.resize(dim);
  reg_.resize(dim);
  row_kind_.assign(static_cast<size_t>(dim), 0);
  diag_slots_.assign(static_cast<size_t>(dim), -1);

  sparsity_ = SparsityMap();
  p_diag_slots_.clear();
  p_offdiag_slots_.clear();
  a_slots_.clear();
  hs_slots_.clear();
  p_diag_value_idx_.assign(static_cast<size_t>(n_), -1);

  for (Index i = 0; i < n_; ++i) {
    p_diag_slots_.push_back(sparsity_.addEntry(i, i));
    diag_slots_[i] = p_diag_slots_.back();
    signs[i] = 1;
  }
  // P: upper triangle in, lower triangle of K out. k is the position in P_upper.valuePtr().
  {
    Index k = 0;
    for (Index c = 0; c < P_upper.outerSize(); ++c) {
      for (SparseMat::InnerIterator it(P_upper, c); it; ++it, ++k) {
        const Index r = static_cast<Index>(it.row());
        if (r == c) p_diag_value_idx_[c] = k;
        if (r < c) p_offdiag_slots_.emplace_back(sparsity_.addEntry(c, r), k);
      }
    }
  }
  {
    Index k = 0;
    for (Index c = 0; c < A.outerSize(); ++c) {
      for (SparseMat::InnerIterator it(A, c); it; ++it, ++k) {
        a_slots_.emplace_back(sparsity_.addEntry(n_ + static_cast<Index>(it.row()), c), k);
      }
    }
  }
  // Hs: dense lower triangle for SOC blocks, diagonal only for Zero/Nonnegative blocks, in the
  // order ConeBase::writeHsLowerTriangle() writes.
  Index max_entries = 0;
  for (Index bi = 0; bi < cones.numBlocks(); ++bi) {
    const ConeType type = cones.blockType(bi);
    const Index off = n_ + cones.blockOffset(bi);
    const bool dense = type == ConeType::SecondOrder;
    const unsigned char kind = type == ConeType::Nonnegative   ? 1
                               : type == ConeType::SecondOrder ? 2
                                                               : 3;
    std::vector<Index> slots;
    for (Index a = 0; a < cones.blockDim(bi); ++a) {
      for (Index b = dense ? 0 : a; b <= a; ++b) {
        slots.push_back(sparsity_.addEntry(off + a, off + b));
      }
      diag_slots_[off + a] = slots.back();
      row_kind_[off + a] = kind;
      if (type == ConeType::Zero) zero_rows.push_back(off + a);
    }
    assert(static_cast<Index>(slots.size()) == cones.numHsEntries(bi));
    max_entries = std::max(max_entries, static_cast<Index>(slots.size()));
    hs_slots_.push_back(std::move(slots));
  }
  hs_scratch_.resize(max_entries);

  K_ = sparsity_.finalize(dim, dim);

  updateSettings(settings);
  if (!ldlt_.analyzePattern(K_, signs)) return false;

  zero_row_pivots_.clear();
  for (Index row : zero_rows) zero_row_pivots_.push_back(ldlt_.iperm()[row]);

  p_outer_ref_.assign(P_upper.outerIndexPtr(), P_upper.outerIndexPtr() + P_upper.outerSize() + 1);
  p_inner_ref_.assign(P_upper.innerIndexPtr(), P_upper.innerIndexPtr() + P_upper.nonZeros());
  a_outer_ref_.assign(A.outerIndexPtr(), A.outerIndexPtr() + A.outerSize() + 1);
  a_inner_ref_.assign(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());
  return writePAValues(&P_upper, &A);
}

void KktSystem::updateSettings(const Settings& settings) {
  const RegularizationSettings& reg = settings.regularization;
  const Scalar base[4] = {reg.static_P, reg.static_nonneg, reg.static_soc, reg.static_zero};
  for (Index i = 0; i < static_cast<Index>(row_kind_.size()); ++i) reg_base_[i] = base[row_kind_[i]];
  reg_proportional_ = reg.static_proportional;
  ldlt_.params.dynamic_enable = reg.dynamic_enable;
  ldlt_.params.dynamic_eps = reg.dynamic_eps;
  ldlt_.params.dynamic_delta = reg.dynamic_delta;
  ldlt_.params.refine_max_iter = settings.refine_max_iter;
  ldlt_.params.refine_reltol = settings.refine_reltol;
  ldlt_.params.refine_abstol = settings.refine_abstol;
  ldlt_.params.refine_stop_ratio = std::max(settings.refine_stop_ratio, Scalar(1));
  ldlt_.params.supernodal = settings.supernodal;  // takes effect at the next analyzePattern()
}

bool KktSystem::samePatterns(const SparseMat* P_upper, const SparseMat* A) const {
  return (!P_upper || samePattern(*P_upper, p_outer_ref_, p_inner_ref_)) &&
         (!A || samePattern(*A, a_outer_ref_, a_inner_ref_));
}

bool KktSystem::writePAValues(const SparseMat* P_upper, const SparseMat* A) {
  if (!samePatterns(P_upper, A)) return false;
  if (P_upper) {
    const Scalar* Px = P_upper->valuePtr();
    for (Index i = 0; i < n_; ++i) {
      const Index k = p_diag_value_idx_[i];
      sparsity_.setValue(K_, p_diag_slots_[i], k >= 0 ? Px[k] : Scalar(0));
    }
    for (const auto& [slot, k] : p_offdiag_slots_) sparsity_.setValue(K_, slot, Px[k]);
  }
  if (A) {
    const Scalar* Ax = A->valuePtr();
    for (const auto& [slot, k] : a_slots_) sparsity_.setValue(K_, slot, Ax[k]);
  }
  return true;
}

bool KktSystem::updateData(const SparseMat* P_upper, const SparseMat* A) {
  if (!writePAValues(P_upper, A)) return false;
  factorized_ = false;
  return true;
}

void KktSystem::writeScaling(const ConeSet& cones) {
  for (Index bi = 0; bi < cones.numBlocks(); ++bi) {
    const std::vector<Index>& slots = hs_slots_[bi];
    const Index ne = static_cast<Index>(slots.size());
    cones.writeHsLowerTriangle(bi, hs_scratch_.data());
    if (cones.blockType(bi) == ConeType::Zero && compliance_.size() == ne) hs_scratch_.head(ne) += compliance_;
    for (Index i = 0; i < ne; ++i) sparsity_.setValue(K_, slots[i], -hs_scratch_[i]);
  }

  Scalar max_diag = 0;
  for (Index slot : diag_slots_) max_diag = std::max(max_diag, std::abs(sparsity_.getValue(K_, slot)));
  reg_.array() = reg_base_.array() + reg_proportional_ * max_diag;
  ldlt_.setStaticRegularization(reg_);
}

bool KktSystem::updateScalingAndFactorize(const ConeSet& cones) {
  writeScaling(cones);
  factorized_ = ldlt_.factorize(K_);
  ++stats_.factorizations;
  stats_.dynamic_reg_pivots += ldlt_.numDynamicRegularized();
  return factorized_;
}

bool KktSystem::updateScalingAndFactorize(const ConeSet& cones, const std::vector<Index>& coulomb_blocks,
                                          const Vec& row0) {
  writeScaling(cones);
  // Lower triangle of an SOC block in row-major order: (0,0) is slot 0, (a,0) slot a(a+1)/2. The
  // lower entries (a,0) keep -Hs(a,0); the diagonal and the transposed entries (0,a) get -r_a.
  Scalar* Kx = K_.valuePtr();
  for (Index bi : coulomb_blocks) {
    const Index p00 = sparsity_.offset(hs_slots_[bi][0]);
    Kx[p00] -= row0[cones.blockOffset(bi)];
  }
  upper_values_.assign(Kx, Kx + K_.nonZeros());
  for (Index bi : coulomb_blocks) {
    const std::vector<Index>& slots = hs_slots_[bi];
    const Index off = cones.blockOffset(bi), d = cones.blockDim(bi);
    for (Index a = 1; a < d; ++a) {
      upper_values_[static_cast<size_t>(sparsity_.offset(slots[static_cast<size_t>(a * (a + 1) / 2)]))] -= row0[off + a];
    }
  }
  factorized_ = ldlt_.factorizeUnsymmetric(K_, upper_values_.data());
  ++stats_.factorizations;
  stats_.dynamic_reg_pivots += ldlt_.numDynamicRegularized();
  return factorized_;
}

Scalar KktSystem::solve(const Vec& rhs, Vec& x_out) {
  assert(factorized_);
  return recordSolve(rhs, ldlt_.solve(rhs, x_out));
}

Scalar KktSystem::solvePair(const Vec& rhs1, Vec& x1, const Vec& rhs2, Vec& x2, bool refine2) {
  assert(factorized_);
  return recordSolve(rhs1, ldlt_.solvePair(rhs1, x1, rhs2, x2, refine2));
}

Scalar KktSystem::recordSolve(const Vec& rhs, Scalar res) {
  stats_.refinement_steps += ldlt_.lastRefineSteps();
  if (res < 0) {  // refinement off: no residual available
    last_refinement_residual_ = 0;
    return 0;
  }
  const Scalar rhs_norm = dim() > 0 ? rhs.lpNorm<Eigen::Infinity>() : Scalar(0);
  last_refinement_residual_ = res / std::max(rhs_norm, Scalar(1e-30));

  if (!zero_row_pivots_.empty()) {
    const Vec& r = ldlt_.lastResidual();
    Scalar r_eq = 0;
    for (Index p : zero_row_pivots_) r_eq = std::max(r_eq, std::abs(r[p]));
    if (r_eq > kEqualityResidualTol * std::max(rhs_norm, Scalar(1))) {
      stats_.equality_rank_deficient = true;
    }
  }
  return last_refinement_residual_;
}

}  // namespace conicxx::detail
