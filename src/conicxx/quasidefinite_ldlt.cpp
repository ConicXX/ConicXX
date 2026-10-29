// The elimination tree and the up-looking numeric factorization follow QDLDL
// (https://github.com/osqp/qdldl, Apache-2.0); see NOTICE.
#include "conicxx/quasidefinite_ldlt.h"

#include <Eigen/OrderingMethods>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace conicxx::detail {

bool QuasiDefiniteLdlt::analyzePattern(const SparseMat& K, const std::vector<signed char>& signs) {
  analyzed_ = false;
  const Index n = static_cast<Index>(K.cols());
  if (K.rows() != K.cols() || static_cast<Index>(signs.size()) != n || !K.isCompressed()) {
    return false;
  }
  n_ = n;

  // Fill-reducing ordering. Eigen's ordering methods return the inverse permutation, whose
  // indices map pivot position -> original row.
  perm_.resize(static_cast<size_t>(n));
  if (n > 0) {
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, Index> pinv;
    Eigen::AMDOrdering<Index> amd;
    amd(K.selfadjointView<Eigen::Lower>(), pinv);
    std::copy(pinv.indices().data(), pinv.indices().data() + n, perm_.begin());
  }
  iperm_.assign(static_cast<size_t>(n), -1);
  for (Index k = 0; k < n; ++k) iperm_[perm_[k]] = k;

  buildPermutedUpper(K);

  // Elimination tree and column counts of L.
  std::vector<Index> etree(n, -1), Lnz(n, 0), mark(n, -1);
  for (Index j = 0; j < n; ++j) {
    mark[j] = j;
    for (Index p = Cp_[j]; p < Cp_[j + 1]; ++p) {
      for (Index i = Ci_[p]; mark[i] != j; i = etree[i]) {
        if (etree[i] == -1) etree[i] = j;
        ++Lnz[i];
        mark[i] = j;
      }
    }
  }
  Lp_.assign(n + 1, 0);
  for (Index j = 0; j < n; ++j) Lp_[j + 1] = Lp_[j] + Lnz[j];
  const Index nnzL = Lp_[n];
  Li_.resize(nnzL);
  Lx_.assign(nnzL, Scalar(0));

  // Row k of L is the reach of column k of C in the elimination tree. Filling rows in the order
  // k = 0, 1, ... leaves every column of L sorted by row.
  std::vector<Index> fill(Lp_.begin(), Lp_.end() - 1);
  std::fill(mark.begin(), mark.end(), -1);
  for (Index k = 0; k < n; ++k) {
    mark[k] = k;
    for (Index p = Cp_[k]; p < Cp_[k + 1]; ++p) {
      for (Index i = Ci_[p]; mark[i] != k; i = etree[i]) {
        mark[i] = k;
        Li_[fill[i]++] = k;
      }
    }
  }

  // Row structure (transposed pattern) with ascending columns, a valid elimination order.
  Rp_.assign(n + 1, 0);
  for (Index p = 0; p < nnzL; ++p) ++Rp_[Li_[p] + 1];
  for (Index k = 0; k < n; ++k) Rp_[k + 1] += Rp_[k];
  Rj_.resize(nnzL);
  Rpos_.resize(nnzL);
  std::vector<Index> rfill(Rp_.begin(), Rp_.end() - 1);
  for (Index j = 0; j < n; ++j) {
    for (Index p = Lp_[j]; p < Lp_[j + 1]; ++p) {
      const Index q = rfill[Li_[p]]++;
      Rj_[q] = j;
      Rpos_[q] = p;
    }
  }

  sign_.resize(n);
  for (Index k = 0; k < n; ++k) sign_[k] = signs[perm_[k]] >= 0 ? Scalar(1) : Scalar(-1);
  reg_.assign(n, Scalar(0));
  D_.assign(n, Scalar(0));
  Dinv_.assign(n, Scalar(0));
  y_.assign(n, Scalar(0));
  bp_.setZero(n);
  xp_.setZero(n);
  xt_.setZero(n);
  r_.setZero(n);
  rt_.setZero(n);
  xp2_.setZero(n);
  r2_.setZero(n);
  y2_.setZero(2 * n);
  num_dyn_reg_ = num_positive_ = 0;
  last_refine_steps_ = 0;
  analyzed_ = true;
  return true;
}

void QuasiDefiniteLdlt::buildPermutedUpper(const SparseMat& K) {
  const Index n = n_;
  const Index* Kp = K.outerIndexPtr();
  const Index* Ki = K.innerIndexPtr();

  Cp_.assign(n + 1, 0);
  for (Index j = 0; j < n; ++j) {
    for (Index p = Kp[j]; p < Kp[j + 1]; ++p) {
      if (Ki[p] >= j) ++Cp_[std::max(iperm_[Ki[p]], iperm_[j]) + 1];
    }
  }
  for (Index j = 0; j < n; ++j) Cp_[j + 1] += Cp_[j];

  Ci_.resize(Cp_[n]);
  Cx_.assign(Cp_[n], Scalar(0));
  KtoC_.assign(Kp[n], -1);
  std::vector<Index> next(Cp_.begin(), Cp_.end() - 1);
  for (Index j = 0; j < n; ++j) {
    for (Index p = Kp[j]; p < Kp[j + 1]; ++p) {
      if (Ki[p] < j) continue;  // upper triangle: ignored
      const Index a = iperm_[Ki[p]], b = iperm_[j];
      const Index q = next[std::max(a, b)]++;
      Ci_[q] = std::min(a, b);
      KtoC_[p] = q;
    }
  }
}

void QuasiDefiniteLdlt::setStaticRegularization(const Vec& reg) {
  assert(reg.size() == n_);
  for (Index k = 0; k < n_; ++k) reg_[k] = reg[perm_[k]];
}

bool QuasiDefiniteLdlt::factorize(const SparseMat& K) {
  if (!analyzed_ || K.nonZeros() != static_cast<Eigen::Index>(KtoC_.size())) return false;
  const Scalar* Kx = K.valuePtr();
  const Index nnzK = static_cast<Index>(KtoC_.size());
  for (Index p = 0; p < nnzK; ++p) {
    if (KtoC_[p] >= 0) Cx_[KtoC_[p]] = Kx[p];
  }

  num_dyn_reg_ = num_positive_ = 0;
  for (Index k = 0; k < n_; ++k) {
    // Scatter column k of C: the strict upper part into y, the diagonal into the pivot.
    Scalar d = sign_[k] * reg_[k];
    for (Index p = Cp_[k]; p < Cp_[k + 1]; ++p) {
      if (Ci_[p] == k) {
        d += Cx_[p];
      } else {
        y_[Ci_[p]] = Cx_[p];
      }
    }
    // Row k of L: solve L(0:k-1, 0:k-1) y = c over the precomputed pattern.
    for (Index q = Rp_[k]; q < Rp_[k + 1]; ++q) {
      const Index j = Rj_[q], pos = Rpos_[q];
      const Scalar yj = y_[j];
      for (Index p = Lp_[j]; p < pos; ++p) y_[Li_[p]] -= Lx_[p] * yj;
      const Scalar l = yj * Dinv_[j];
      Lx_[pos] = l;
      d -= yj * l;
      y_[j] = Scalar(0);
    }
    if (params.dynamic_enable && d * sign_[k] < params.dynamic_eps) {
      d = sign_[k] * params.dynamic_delta;
      ++num_dyn_reg_;
    }
    if (!std::isfinite(d) || d == Scalar(0)) {
      std::fill(y_.begin(), y_.end(), Scalar(0));
      return false;
    }
    if (d > 0) ++num_positive_;
    D_[k] = d;
    Dinv_[k] = Scalar(1) / d;
  }
  return true;
}

void QuasiDefiniteLdlt::solveInPlace(Vec& x) const {
  for (Index j = 0; j < n_; ++j) {
    const Scalar xj = x[j];
    if (xj == Scalar(0)) continue;
    for (Index p = Lp_[j]; p < Lp_[j + 1]; ++p) x[Li_[p]] -= Lx_[p] * xj;
  }
  for (Index j = 0; j < n_; ++j) x[j] *= Dinv_[j];
  for (Index j = n_ - 1; j >= 0; --j) {
    Scalar s = x[j];
    for (Index p = Lp_[j]; p < Lp_[j + 1]; ++p) s -= Lx_[p] * x[Li_[p]];
    x[j] = s;
  }
}

void QuasiDefiniteLdlt::solvePairInPlace(Vec& y) const {
  for (Index j = 0; j < n_; ++j) {
    const Scalar a = y[2 * j], b = y[2 * j + 1];
    if (a == Scalar(0) && b == Scalar(0)) continue;
    for (Index p = Lp_[j]; p < Lp_[j + 1]; ++p) {
      const Index i = 2 * Li_[p];
      y[i] -= Lx_[p] * a;
      y[i + 1] -= Lx_[p] * b;
    }
  }
  for (Index j = 0; j < n_; ++j) {
    y[2 * j] *= Dinv_[j];
    y[2 * j + 1] *= Dinv_[j];
  }
  for (Index j = n_ - 1; j >= 0; --j) {
    Scalar a = y[2 * j], b = y[2 * j + 1];
    for (Index p = Lp_[j]; p < Lp_[j + 1]; ++p) {
      const Index i = 2 * Li_[p];
      a -= Lx_[p] * y[i];
      b -= Lx_[p] * y[i + 1];
    }
    y[2 * j] = a;
    y[2 * j + 1] = b;
  }
}

Scalar QuasiDefiniteLdlt::residual(const Vec& x, Vec& r) const {
  r = bp_;
  for (Index j = 0; j < n_; ++j) {
    const Scalar xj = x[j];
    Scalar s = 0;
    for (Index p = Cp_[j]; p < Cp_[j + 1]; ++p) {
      const Index i = Ci_[p];
      r[i] -= Cx_[p] * xj;
      if (i != j) s += Cx_[p] * x[i];
    }
    r[j] -= s;
  }
  return n_ > 0 ? r.lpNorm<Eigen::Infinity>() : Scalar(0);
}

Scalar QuasiDefiniteLdlt::refine(Vec& xp, Vec& r, int& steps) {
  // Iterative refinement on the exact matrix against bp_: accept a correction only if it reduces
  // the residual, and stop when it no longer shrinks by refine_stop_ratio per step. Without
  // refinement no residual is computed (returns -1).
  if (params.refine_max_iter <= 0) return Scalar(-1);
  Scalar norm = residual(xp, r);
  const Scalar tol =
      params.refine_abstol + params.refine_reltol * (n_ > 0 ? bp_.lpNorm<Eigen::Infinity>() : 0);
  for (int it = 0; it < params.refine_max_iter && norm > tol; ++it) {
    xt_ = r;
    solveInPlace(xt_);
    xt_ += xp;
    const Scalar trial = residual(xt_, rt_);
    ++steps;
    if (!(trial < norm)) break;
    xp.swap(xt_);
    r.swap(rt_);
    const bool fast = norm >= params.refine_stop_ratio * trial;
    norm = trial;
    if (!fast) break;
  }
  return norm;
}

Scalar QuasiDefiniteLdlt::solve(const Vec& b, Vec& x) {
  assert(b.size() == n_);
  if (x.size() != n_) x.resize(n_);
  for (Index k = 0; k < n_; ++k) bp_[k] = b[perm_[k]];
  xp_ = bp_;
  solveInPlace(xp_);
  last_refine_steps_ = 0;
  const Scalar norm = refine(xp_, r_, last_refine_steps_);
  for (Index k = 0; k < n_; ++k) x[perm_[k]] = xp_[k];
  return norm;
}

Scalar QuasiDefiniteLdlt::solvePair(const Vec& b1, Vec& x1, const Vec& b2, Vec& x2, bool refine2) {
  assert(b1.size() == n_ && b2.size() == n_);
  if (x1.size() != n_) x1.resize(n_);
  if (x2.size() != n_) x2.resize(n_);
  for (Index k = 0; k < n_; ++k) {
    y2_[2 * k] = b1[perm_[k]];
    y2_[2 * k + 1] = b2[perm_[k]];
  }
  solvePairInPlace(y2_);
  last_refine_steps_ = 0;
  if (refine2) {
    for (Index k = 0; k < n_; ++k) {
      bp_[k] = b2[perm_[k]];
      xp2_[k] = y2_[2 * k + 1];
    }
    refine(xp2_, r2_, last_refine_steps_);
    for (Index k = 0; k < n_; ++k) x2[perm_[k]] = xp2_[k];
  } else {
    for (Index k = 0; k < n_; ++k) x2[perm_[k]] = y2_[2 * k + 1];
  }
  for (Index k = 0; k < n_; ++k) {
    bp_[k] = b1[perm_[k]];
    xp_[k] = y2_[2 * k];
  }
  const Scalar norm = refine(xp_, r_, last_refine_steps_);
  for (Index k = 0; k < n_; ++k) x1[perm_[k]] = xp_[k];
  return norm;
}

}  // namespace conicxx::detail
