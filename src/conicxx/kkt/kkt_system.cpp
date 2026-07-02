#include "conicxx/kkt/kkt_system.h"

#include <algorithm>
#include <cmath>

namespace conicxx::detail {

namespace {
bool sameSparsityPattern(const SparseMat& mat, const std::vector<Index>& outer_ref,
                          const std::vector<Index>& inner_ref) {
  if (mat.outerSize() + 1 != static_cast<Index>(outer_ref.size())) return false;
  if (!std::equal(mat.outerIndexPtr(), mat.outerIndexPtr() + mat.outerSize() + 1,
                   outer_ref.begin())) {
    return false;
  }
  if (mat.nonZeros() != static_cast<Index>(inner_ref.size())) return false;
  return std::equal(mat.innerIndexPtr(), mat.innerIndexPtr() + mat.nonZeros(), inner_ref.begin());
}
}  // namespace

bool KktSystem::setup(const SparseMat& P_upper, const SparseMat& A, const ConeSet& cones,
                      const Settings& settings) {
  n_ = static_cast<Index>(P_upper.rows());
  m_ = cones.totalDim();
  static_reg_p_ = settings.static_reg_P;
  static_reg_a_ = settings.static_reg_A;
  dynamic_reg_eps_ = settings.dynamic_reg_eps;
  dynamic_reg_delta_ = settings.dynamic_reg_delta;
  refine_max_iter_ = settings.refine_max_iter;
  refine_tol_ = settings.refine_tol;
  factorized_ = false;

  sparsity_ = SparsityMap();
  p_diag_slots_.clear();
  p_diag_slots_.reserve(static_cast<size_t>(n_));
  p_offdiag_slots_.clear();
  a_slots_.clear();
  hs_blocks_.clear();

  for (Index i = 0; i < n_; ++i) {
    p_diag_slots_.push_back(sparsity_.addEntry(i, i));
  }

  for (Index c = 0; c < P_upper.outerSize(); ++c) {
    for (SparseMat::InnerIterator it(P_upper, c); it; ++it) {
      const Index r = static_cast<Index>(it.row());
      if (r >= c) continue;  // diagonal handled above; input assumed upper-triangular
      const Index slot = sparsity_.addEntry(c, r);
      p_offdiag_slots_.emplace_back(slot, r, c);
    }
  }

  for (Index c = 0; c < A.outerSize(); ++c) {
    for (SparseMat::InnerIterator it(A, c); it; ++it) {
      const Index r = static_cast<Index>(it.row());
      const Index slot = sparsity_.addEntry(n_ + r, c);
      a_slots_.emplace_back(slot, r, c);
    }
  }

  for (Index bi = 0; bi < cones.numBlocks(); ++bi) {
    const ConeBase& blk = cones.block(bi);
    const Index off = cones.blockOffset(bi);
    const Index d = blk.dim();
    HsBlockSlots hbs;
    hbs.z_offset = off;
    hbs.dim = d;
    if (blk.type() == ConeType::SecondOrder) {
      // The NT scaling block Hs = W^T W is genuinely dense for SOC blocks.
      for (Index a = 0; a < d; ++a) {
        for (Index b = 0; b <= a; ++b) {
          const Index slot = sparsity_.addEntry(n_ + off + a, n_ + off + b);
          hbs.slots.emplace_back(slot, a, b);
        }
      }
    } else {
      // Zero/Nonnegative cones: Hs is (at most) diagonal, so only the
      // diagonal needs a slot in K (it also carries static/dynamic
      // regularization). Registering the full triangle here would
      // explicitly store an always-zero-valued dense block in K's
      // sparsity pattern -- for a large equality (Zero-cone) or
      // nonnegative block this causes catastrophic fill-in during
      // symbolic/numeric factorization for a block that mathematically
      // contributes nothing off-diagonal.
      for (Index a = 0; a < d; ++a) {
        const Index slot = sparsity_.addEntry(n_ + off + a, n_ + off + a);
        hbs.slots.emplace_back(slot, a, a);
      }
    }
    hs_blocks_.push_back(std::move(hbs));
  }

  K_ = sparsity_.finalize(n_ + m_, n_ + m_);
  ldlt_.analyzePattern(K_);

  p_outer_ref_.assign(P_upper.outerIndexPtr(), P_upper.outerIndexPtr() + P_upper.outerSize() + 1);
  p_inner_ref_.assign(P_upper.innerIndexPtr(), P_upper.innerIndexPtr() + P_upper.nonZeros());
  a_outer_ref_.assign(A.outerIndexPtr(), A.outerIndexPtr() + A.outerSize() + 1);
  a_inner_ref_.assign(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());

  if (!updateData(&P_upper, &A)) return false;
  return updateScalingAndFactorize(cones);
}

bool KktSystem::updateData(const SparseMat* P_upper, const SparseMat* A) {
  if (P_upper) {
    if (!sameSparsityPattern(*P_upper, p_outer_ref_, p_inner_ref_)) return false;
    for (Index i = 0; i < n_; ++i) {
      sparsity_.setValue(K_, p_diag_slots_[static_cast<size_t>(i)],
                          P_upper->coeff(i, i) + static_reg_p_);
    }
    for (const auto& [slot, r, c] : p_offdiag_slots_) {
      sparsity_.setValue(K_, slot, P_upper->coeff(r, c));
    }
  }
  if (A) {
    if (!sameSparsityPattern(*A, a_outer_ref_, a_inner_ref_)) return false;
    for (const auto& [slot, r, c] : a_slots_) {
      sparsity_.setValue(K_, slot, A->coeff(r, c));
    }
  }
  return factorizeWithRetry();
}

bool KktSystem::updateScalingAndFactorize(const ConeSet& cones) {
  const auto& blocks = cones.scalingBlocks();
  for (size_t bi = 0; bi < hs_blocks_.size(); ++bi) {
    const Mat& Hs = blocks[bi];
    for (const auto& [slot, a, b] : hs_blocks_[bi].slots) {
      Scalar v = -Hs(a, b);
      if (a == b) v -= static_reg_a_;
      sparsity_.setValue(K_, slot, v);
    }
  }
  return factorizeWithRetry();
}

void KktSystem::writeRegularizedDiagonal(SparseMat& K, Scalar extra_p_reg,
                                         Scalar extra_a_reg) const {
  if (extra_p_reg != 0.0) {
    for (Index slot : p_diag_slots_) {
      sparsity_.setValue(K, slot, sparsity_.getValue(K, slot) + extra_p_reg);
    }
  }
  if (extra_a_reg != 0.0) {
    for (const auto& hbs : hs_blocks_) {
      for (const auto& [slot, a, b] : hbs.slots) {
        if (a == b) {
          sparsity_.setValue(K, slot, sparsity_.getValue(K, slot) - extra_a_reg);
        }
      }
    }
  }
}

bool KktSystem::factorizeWithRetry() {
  Scalar extra_p = 0.0, extra_a = 0.0;
  constexpr int kMaxRetries = 8;
  for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
    // Pattern is fixed after setup()'s one-time analyzePattern() call, so
    // only factorize() (numeric refactorization) is needed here -- never
    // compute(), which would redo the fill-reducing symbolic analysis from
    // scratch on every IPM iteration. The K_ copy is only needed when a
    // dynamic-regularization bump must be applied on top of it; the common
    // (no-retry) case factorizes K_ directly.
    if (extra_p != 0.0 || extra_a != 0.0) {
      SparseMat K_try = K_;
      writeRegularizedDiagonal(K_try, extra_p, extra_a);
      ldlt_.factorize(K_try);
    } else {
      ldlt_.factorize(K_);
    }
    if (ldlt_.info() == Eigen::Success) {
      const Vec D = ldlt_.vectorD();
      const Scalar minAbsD = D.size() > 0 ? D.array().abs().minCoeff() : Scalar(1.0);
      if (std::isfinite(minAbsD) && minAbsD > dynamic_reg_eps_) {
        factorized_ = true;
        return true;
      }
    }
    extra_p = std::max(extra_p * 10.0, dynamic_reg_delta_);
    extra_a = std::max(extra_a * 10.0, dynamic_reg_delta_);
  }
  factorized_ = false;
  return false;
}

Scalar KktSystem::solve(const Vec& rhs, Vec& x_out) const {
  x_out = ldlt_.solve(rhs);

  const Scalar rhsNorm = std::max(rhs.norm(), Scalar(1e-30));
  Vec r = rhs - K_.selfadjointView<Eigen::Lower>() * x_out;
  Scalar relRes = r.norm() / rhsNorm;

  for (int it = 0; it < refine_max_iter_ && relRes > refine_tol_; ++it) {
    Vec dx = ldlt_.solve(r);
    x_out += dx;
    r = rhs - K_.selfadjointView<Eigen::Lower>() * x_out;
    relRes = r.norm() / rhsNorm;
  }
  return relRes;
}

}  // namespace conicxx::detail
