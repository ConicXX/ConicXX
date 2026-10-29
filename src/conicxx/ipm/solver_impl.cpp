#include "conicxx/ipm/solver_impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace conicxx::detail {

namespace {
bool validCompliance(const Vec* c, Index zero_dim) {
  return !c || (c->size() == zero_dim && c->allFinite() && (c->array() >= 0).all());
}

bool finiteData(const SparseMat* P, const Vec* q, const SparseMat* A, const Vec* b) {
  const auto finiteValues = [](const SparseMat* M) {
    if (!M) return true;
    for (Index c = 0; c < M->outerSize(); ++c) {
      for (SparseMat::InnerIterator it(*M, c); it; ++it) {
        if (!std::isfinite(it.value())) return false;
      }
    }
    return true;
  };
  return finiteValues(P) && finiteValues(A) && (!q || q->allFinite()) && (!b || b->allFinite());
}
}  // namespace

bool SolverImpl::setup(const SparseMat& P, const Vec& q, const SparseMat& A, const Vec& b,
                       const ConeSpec& cone_spec, const Settings& settings,
                       const Vec* zero_compliance) {
  const auto setup_start = std::chrono::steady_clock::now();
  setup_done_ = false;
  if (!cone_spec.isValid()) return false;
  if (P.rows() != P.cols()) return false;

  n_ = static_cast<Index>(P.rows());
  m_ = cone_spec.totalDim();
  if (A.rows() != m_ || A.cols() != n_ || b.size() != m_ || q.size() != n_) return false;
  if (settings.validate_inputs && !finiteData(&P, &q, &A, &b)) return false;
  if (!validCompliance(zero_compliance, cone_spec.zero_dim)) return false;

  cone_spec_ = cone_spec;
  settings_ = settings;
  cones_ = std::make_unique<ConeSet>(cone_spec_);

  P_ = P;
  q_ = q;
  A_ = A;
  b_ = b;
  P_.makeCompressed();
  A_.makeCompressed();

  equil_ = Equilibration();
  equilibrated_ = settings_.equilibrate;
  if (equilibrated_) {
    equil_.compute(P_, q_, A_, b_, zero_compliance ? *zero_compliance : Vec(), *cones_, settings_);
    equil_.scaleProblem(P_, q_, A_, b_);
  }

  if (!kkt_.setup(P_, A_, *cones_, settings_)) return false;
  setCompliance(zero_compliance);
  Cz_ = Vec::Zero(cone_spec_.zero_dim);

  cacheEquilibrationWeights();
  cacheUnscaledDataNorms(/*b_changed=*/true, /*q_changed=*/true);

  x_ = Vec::Zero(n_);
  s_ = Vec::Zero(m_);
  z_ = Vec::Zero(m_);
  Px_ = Vec::Zero(n_);
  rx_ = Vec::Zero(n_);
  rz_ = Vec::Zero(m_);
  x1_ = Vec::Zero(n_);
  z1_ = Vec::Zero(m_);
  Px1_ = Vec::Zero(n_);
  dx_aff_ = Vec::Zero(n_);
  dz_aff_ = Vec::Zero(m_);
  ds_aff_ = Vec::Zero(m_);
  dx_ = Vec::Zero(n_);
  dz_ = Vec::Zero(m_);
  ds_ = Vec::Zero(m_);

  kkt_rhs_.resize(n_ + m_);
  kkt_sol_.resize(n_ + m_);
  kkt_rhs_aff_.resize(n_ + m_);
  kkt_sol_aff_.resize(n_ + m_);
  Pmul_scratch_.resize(n_);
  xi_scratch_.resize(n_);
  xi_minus_x1_scratch_.resize(n_);
  Hz_scratch_.resize(m_);
  Atz_scratch_.resize(n_);
  Ax_scratch_.resize(m_);
  dz_table_.resize(m_);
  lambda_.resize(m_);
  lambda_prod_.resize(m_);
  Winv_ds_aff_.resize(m_);
  W_dz_aff_.resize(m_);
  corrector_.resize(m_);
  ds_combined_.resize(m_);
  lambda_inv_ds_.resize(m_);
  ds_const_.resize(m_);
  s_trial_.resize(m_);
  z_trial_.resize(m_);
  lambda_trial_.resize(m_);
  rx_inf_.resize(n_);
  rz_inf_.resize(m_);
  s_aff_scratch_.resize(m_);
  z_aff_scratch_.resize(m_);
  warm_x_trial_.resize(n_);
  warm_s_trial_.resize(m_);
  warm_z_trial_.resize(m_);
  sol_x_scratch_.resize(n_);
  sol_s_scratch_.resize(m_);
  sol_z_scratch_.resize(m_);

  have_warm_start_ = false;
  have_cold_merit_ = false;
  solution_ = Solution{};
  setup_done_ = true;
  setup_time_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - setup_start).count();
  return true;
}

bool SolverImpl::updateData(const SparseMat* P, const Vec* q, const SparseMat* A, const Vec* b,
                            const Vec* zero_compliance) {
  if (!setup_done_) return false;
  if (settings_.validate_inputs && !finiteData(P, q, A, b)) return false;
  if (!validCompliance(zero_compliance, cone_spec_.zero_dim)) return false;
  if (P && (P->rows() != n_ || P->cols() != n_)) return false;
  if (A && (A->rows() != m_ || A->cols() != n_)) return false;
  if (q && q->size() != n_) return false;
  if (b && b->size() != m_) return false;
  if (!kkt_.samePatterns(P, A)) return false;

  // Same patterns: copy the values into the stored data and scale them in place.
  if (P) {
    std::copy(P->valuePtr(), P->valuePtr() + P->nonZeros(), P_.valuePtr());
    if (equilibrated_) equil_.scaleP(P_);
  }
  if (A) {
    std::copy(A->valuePtr(), A->valuePtr() + A->nonZeros(), A_.valuePtr());
    if (equilibrated_) equil_.scaleA(A_);
  }
  if (q) {
    q_ = *q;
    if (equilibrated_) equil_.scaleQ(q_);
  }
  if (b) {
    b_ = *b;
    if (equilibrated_) equil_.scaleB(b_);
  }
  kkt_.updateData(P ? &P_ : nullptr, A ? &A_ : nullptr);
  if (zero_compliance) setCompliance(zero_compliance);

  cacheUnscaledDataNorms(b != nullptr, q != nullptr);
  return true;
}

void SolverImpl::setCompliance(const Vec* zero_compliance) {
  if (zero_compliance) {
    compliance_ = *zero_compliance;
    if (equilibrated_) equil_.scaleCompliance(compliance_);
  } else {
    compliance_.setZero(cone_spec_.zero_dim);
  }
  has_compliance_ = compliance_.size() > 0 && compliance_.maxCoeff() > 0;
  kkt_.setZeroCompliance(compliance_);
}

void SolverImpl::setWarmStart(const Vec& x, const Vec& s, const Vec& z) {
  if (!setup_done_ || x.size() != n_ || s.size() != m_ || z.size() != m_) return;
  if (!x.allFinite() || !s.allFinite() || !z.allFinite()) return;
  warm_x_ = x;
  warm_s_ = s;
  warm_z_ = z;
  if (equilibrated_) equil_.scaleSolution(warm_x_, warm_s_, warm_z_);
  have_warm_start_ = true;
}

void SolverImpl::cacheEquilibrationWeights() {
  // Always valid regardless of equilibrated_: d_eff_/e_eff_ are the real Ruiz weights
  // when equilibration ran, or empty (meaning "identity", see weightedInfNorm) when it didn't --
  // every call site goes through weightedInfNorm() so this is the only place that branches on
  // equilibrated_ for this purpose.
  if (equilibrated_) {
    d_eff_ = equil_.d();
    e_eff_ = equil_.e();
    dinv_eff_ = d_eff_.cwiseInverse();
    einv_eff_ = e_eff_.cwiseInverse();
    cinv_ = 1.0 / equil_.c();
  } else {
    d_eff_ = Vec();
    e_eff_ = Vec();
    dinv_eff_ = Vec();
    einv_eff_ = Vec();
    cinv_ = 1.0;
  }
}

Scalar SolverImpl::weightedInfNorm(const Eigen::Ref<const Vec>& v, const Vec& weights) {
  if (v.size() == 0) return 0.0;
  if (weights.size() == 0) return v.cwiseAbs().maxCoeff();  // equilibrate == false: identity
  return (v.array() * weights.head(v.size()).array()).abs().maxCoeff();
}

void SolverImpl::cacheUnscaledDataNorms(bool b_changed, bool q_changed) {
  // ||b||_inf = ||E^-1 b_hat||_inf (b_ is b_hat, the stored equilibrated value -- see
  // equilibration.h: b_hat = E*b). ||q||_inf = ||(1/c) D^-1 q_hat||_inf likewise.
  if (b_changed) normb_ = weightedInfNorm(b_, einv_eff_);
  if (q_changed) normq_ = cinv_ * weightedInfNorm(q_, dinv_eff_);
}

void SolverImpl::shiftToInteriorCold(Eigen::Ref<Vec> v) const {
  const auto [min_margin, pos_margin] = cones_->margins(v);
  const Scalar deg = static_cast<Scalar>(std::max(cones_->degree(), Index(1)));
  const Scalar target = std::max(Scalar(1.0), Scalar(0.1) * pos_margin / deg);
  if (min_margin <= 0.0) {
    cones_->scaledUnitShift(v, -min_margin);
    cones_->scaledUnitShift(v, target);
  } else if (min_margin < target) {
    cones_->scaledUnitShift(v, target - min_margin);
  } else {
    cones_->scaledUnitShift(v, 0.0);
  }
}

void SolverImpl::recenterWarmStart(Eigen::Ref<Vec> x, Eigen::Ref<Vec> s, Eigen::Ref<Vec> z) const {
  const Scalar lambda = std::clamp(settings_.warm_lambda, Scalar(0), Scalar(1) - Scalar(1e-12));
  x *= lambda;
  s *= lambda;
  z *= lambda;
  cones_->scaledUnitShift(s, 1.0 - lambda);
  cones_->scaledUnitShift(z, 1.0 - lambda);
}

bool SolverImpl::computeInitialPoint() {
  cones_->updateScaling(cones_->identityElement(), cones_->identityElement());
  if (!kkt_.updateScalingAndFactorize(*cones_)) return false;

  if (P_.nonZeros() == 0 && !has_compliance_) {
    kkt_rhs_.head(n_).setZero();
    kkt_rhs_.tail(m_) = b_;
    kkt_.solve(kkt_rhs_, kkt_sol_);
    x_ = kkt_sol_.head(n_);
    s_ = -kkt_sol_.tail(m_);

    kkt_rhs_.head(n_) = -q_;
    kkt_rhs_.tail(m_).setZero();
    kkt_.solve(kkt_rhs_, kkt_sol_);
    z_ = kkt_sol_.tail(m_);
  } else {
    kkt_rhs_.head(n_) = -q_;
    kkt_rhs_.tail(m_) = b_;
    kkt_.solve(kkt_rhs_, kkt_sol_);
    x_ = kkt_sol_.head(n_);
    z_ = kkt_sol_.tail(m_);
    s_ = -z_;
  }

  cones_->zeroPrimalZeroConeBlocks(s_);
  shiftToInteriorCold(s_);
  shiftToInteriorCold(z_);

  tau_ = 1.0;
  kappa_ = 1.0;
  return x_.allFinite() && s_.allFinite() && z_.allFinite();
}

bool SolverImpl::refactorizeForCurrentScaling() {
  cones_->updateScaling(s_, z_);
  return kkt_.updateScalingAndFactorize(*cones_);
}

bool SolverImpl::computeConstantSolve() {
  // The constant right-hand side (-q, b) and the affine one (rx, s - rz) share one pass over the
  // factor; computeAffineStep() picks up the second solution from kkt_sol_aff_.
  kkt_rhs_.head(n_) = -q_;
  kkt_rhs_.tail(m_) = b_;
  kkt_rhs_aff_.head(n_) = rx_;
  kkt_rhs_aff_.tail(m_) = s_ - rz_;
  kkt_.solvePair(kkt_rhs_, kkt_sol_, kkt_rhs_aff_, kkt_sol_aff_, settings_.refine_affine);
  x1_ = kkt_sol_.head(n_);
  z1_ = kkt_sol_.tail(m_);
  Pmul(x1_, Px1_);
  return x1_.allFinite() && z1_.allFinite();
}

void SolverImpl::computeResiduals() {
  Pmul(x_, Px_);
  Atz_scratch_.noalias() = A_.transpose() * z_;
  rx_ = -Atz_scratch_ - Px_ - tau_ * q_;
  Ax_scratch_.noalias() = A_ * x_;
  rz_ = Ax_scratch_ + s_ - tau_ * b_;
  dot_qx_ = q_.dot(x_);
  dot_bz_ = b_.dot(z_);
  dot_sz_ = s_.dot(z_);
  dot_xPx_ = x_.dot(Px_);
  if (has_compliance_) {
    const Index nz = cone_spec_.zero_dim;
    Cz_ = compliance_.cwiseProduct(z_.head(nz));
    rz_.head(nz) -= Cz_;
    dot_xPx_ += z_.head(nz).dot(Cz_);  // pi'C pi with pi = -z_E
  }
  rtau_ = dot_qx_ + dot_bz_ + kappa_ + dot_xPx_ / tau_;
}

Scalar SolverImpl::computeMu() const {
  return (dot_sz_ + tau_ * kappa_) / static_cast<Scalar>(cones_->degree() + 1);
}

Scalar SolverImpl::complianceCross(const Eigen::Ref<const Vec>& zv) const {
  // zeta'C zv_E with zeta = z_E / tau: the pi part of xi'P xv in the tau elimination.
  if (!has_compliance_) return 0;
  return Cz_.dot(zv.head(cone_spec_.zero_dim)) / tau_;
}

Scalar SolverImpl::complianceDen() const {
  // (zeta - z1_E)'C(zeta - z1_E) - z1_E'C z1_E = zeta'C zeta - 2 zeta'C z1_E.
  if (!has_compliance_) return 0;
  const auto zE = z_.head(cone_spec_.zero_dim);
  return (zE.dot(Cz_) / tau_ - 2.0 * Cz_.dot(z1_.head(cone_spec_.zero_dim))) / tau_;
}

bool SolverImpl::computeAffineStep() {
  // Two solves feed the tau/kappa elimination: the "variable" solve (xv,zv, RHS (rx, s - rz),
  // solved in computeConstantSolve() together with the constant one) and the "constant" solve
  // (x1_,z1_, RHS=(-q,b), cached once per iteration by computeConstantSolve()). tau_num uses the VARIABLE solve, tau_den uses
  // the CONSTANT one -- this maps onto Clarabel's kktsystem.rs exactly,
  // where (confusingly) `self.x1/z1` denotes the per-call variable solve
  // and `self.x2/z2` denotes the constant one.
  const auto xv = kkt_sol_aff_.head(n_);
  const auto zv = kkt_sol_aff_.tail(m_);
  if (!xv.allFinite() || !zv.allFinite()) return false;

  const Scalar dtau_rhs = rtau_;
  const Scalar dkappa_rhs = tau_ * kappa_;
  xi_scratch_ = x_ / tau_;

  Pmul(xv, Pmul_scratch_);
  xi_minus_x1_scratch_ = xi_scratch_ - x1_;
  const Scalar xi_P_xv = xi_scratch_.dot(Pmul_scratch_);
  Pmul(xi_minus_x1_scratch_, Pmul_scratch_);
  const Scalar tau_num = dtau_rhs - dkappa_rhs / tau_ + q_.dot(xv) + b_.dot(zv) +
                          2.0 * (xi_P_xv + complianceCross(zv));
  const Scalar tau_den = kappa_ / tau_ - q_.dot(x1_) - b_.dot(z1_) +
                          xi_minus_x1_scratch_.dot(Pmul_scratch_) - x1_.dot(Px1_) +
                          complianceDen();
  if (tau_den == 0.0 || !std::isfinite(tau_num) || !std::isfinite(tau_den)) return false;

  dtau_aff_ = tau_num / tau_den;
  dx_aff_ = xv + dtau_aff_ * x1_;
  dz_aff_ = zv + dtau_aff_ * z1_;

  cones_->mulHs(dz_aff_, Hz_scratch_);
  ds_aff_ = -(Hz_scratch_ + s_);

  dkappa_aff_ = -(dkappa_rhs + kappa_ * dtau_aff_) / tau_;

  return std::isfinite(dtau_aff_) && std::isfinite(dkappa_aff_) && dx_aff_.allFinite() &&
         dz_aff_.allFinite() && ds_aff_.allFinite();
}

bool SolverImpl::computeCombinedStep(Scalar sigma, Scalar mu, Scalar corrector_scale) {
  dz_table_ = (1.0 - sigma) * rz_;

  cones_->applyW(z_, lambda_);
  cones_->product(lambda_, lambda_, lambda_prod_);

  cones_->applyWInv(ds_aff_, Winv_ds_aff_);
  cones_->applyW(dz_aff_, W_dz_aff_);
  cones_->product(Winv_ds_aff_, W_dz_aff_, corrector_);
  if (corrector_scale != 1.0) corrector_ *= corrector_scale;

  ds_combined_ = lambda_prod_ + corrector_ - sigma * mu * cones_->identityElement();

  cones_->inverseProduct(lambda_, ds_combined_, lambda_inv_ds_);
  cones_->applyW(lambda_inv_ds_, ds_const_);

  kkt_rhs_.head(n_) = (1.0 - sigma) * rx_;
  kkt_rhs_.tail(m_) = ds_const_ - dz_table_;
  kkt_.solve(kkt_rhs_, kkt_sol_);
  const auto xv = kkt_sol_.head(n_);
  const auto zv = kkt_sol_.tail(m_);
  if (!xv.allFinite() || !zv.allFinite()) return false;

  const Scalar dtau_rhs = (1.0 - sigma) * rtau_;
  const Scalar dkappa_rhs = tau_ * kappa_ + corrector_scale * dtau_aff_ * dkappa_aff_ - sigma * mu;

  xi_scratch_ = x_ / tau_;
  Pmul(xv, Pmul_scratch_);
  xi_minus_x1_scratch_ = xi_scratch_ - x1_;
  const Scalar xi_P_xv = xi_scratch_.dot(Pmul_scratch_);
  Pmul(xi_minus_x1_scratch_, Pmul_scratch_);
  const Scalar tau_num = dtau_rhs - dkappa_rhs / tau_ + q_.dot(xv) + b_.dot(zv) +
                          2.0 * (xi_P_xv + complianceCross(zv));
  const Scalar tau_den = kappa_ / tau_ - q_.dot(x1_) - b_.dot(z1_) +
                          xi_minus_x1_scratch_.dot(Pmul_scratch_) - x1_.dot(Px1_) +
                          complianceDen();
  if (tau_den == 0.0 || !std::isfinite(tau_num) || !std::isfinite(tau_den)) return false;

  dtau_ = tau_num / tau_den;
  dx_ = xv + dtau_ * x1_;
  dz_ = zv + dtau_ * z1_;

  cones_->mulHs(dz_, Hz_scratch_);
  ds_ = -(Hz_scratch_ + ds_const_);

  dkappa_ = -(dkappa_rhs + kappa_ * dtau_) / tau_;

  return std::isfinite(dtau_) && std::isfinite(dkappa_) && dx_.allFinite() && dz_.allFinite() &&
         ds_.allFinite();
}

Scalar SolverImpl::computeStepLength(const Vec& ds, const Vec& dz, Scalar dtau,
                                     Scalar dkappa) const {
  Scalar alpha = 1.0;
  if (dtau < 0) alpha = std::min(alpha, -tau_ / dtau);
  if (dkappa < 0) alpha = std::min(alpha, -kappa_ / dkappa);
  alpha = cones_->maxStep(z_, dz, alpha);
  alpha = cones_->maxStep(s_, ds, alpha);
  return alpha;
}

Scalar SolverImpl::safeguardedStepLength(Scalar alpha_max) const {
  // Backtrack until the trial point is strictly interior and every complementarity pair (each cone
  // block and tau*kappa) has centrality >= centrality_theta * mu_trial. The centrality is evaluated
  // exactly at the trial point (ConeBase::centrality), not through the current scaling W.
  const Scalar deg1 = static_cast<Scalar>(cones_->degree() + 1);
  const bool has_cones = cones_->degree() > 0;
  Scalar alpha = alpha_max;
  constexpr int kMaxBacktracks = 64;
  for (int attempt = 0; attempt < kMaxBacktracks; ++attempt, alpha *= settings_.linesearch_backtrack) {
    const Scalar tau_trial = tau_ + alpha * dtau_;
    const Scalar kappa_trial = kappa_ + alpha * dkappa_;
    if (!(tau_trial > 0.0) || !(kappa_trial > 0.0)) continue;
    s_trial_ = s_ + alpha * ds_;
    z_trial_ = z_ + alpha * dz_;
    if (has_cones && (!(cones_->margins(s_trial_).first > 0.0) || !(cones_->margins(z_trial_).first > 0.0))) {
      continue;
    }
    const Scalar mu_trial = (s_trial_.dot(z_trial_) + tau_trial * kappa_trial) / deg1;
    const Scalar bound = settings_.centrality_theta * mu_trial;
    if (tau_trial * kappa_trial >= bound && cones_->minCentrality(s_trial_, z_trial_) >= bound) {
      return alpha;
    }
  }
  return alpha;
}

void SolverImpl::addStep(Scalar alpha) {
  x_ += alpha * dx_;
  s_ += alpha * ds_;
  z_ += alpha * dz_;
  tau_ += alpha * dtau_;
  kappa_ += alpha * dkappa_;
}

void SolverImpl::maybeRescale() {
  const Scalar scale = std::max(tau_, kappa_);
  if (scale > 1e8 || scale < 1e-8) {
    const Scalar inv = 1.0 / scale;
    x_ *= inv;
    s_ *= inv;
    z_ *= inv;
    tau_ *= inv;
    kappa_ *= inv;
  }
}

SolverImpl::Metrics SolverImpl::computeMetrics() const {
  Metrics m;
  const Scalar tinv = 1.0 / tau_;

  // "Direct" (not tau-normalized) unscaled norms: the infeasibility certificates (T4.2) evaluate
  // the homogeneous x_/z_/s_ themselves, not x_/tau_ etc. -- a certificate is meaningful exactly
  // when tau -> 0, where dividing by tau would blow up for no reason.
  const Scalar normx_direct = weightedInfNorm(x_, d_eff_);
  const Scalar normz_direct = cinv_ * weightedInfNorm(z_, e_eff_);
  const Scalar norms_direct = weightedInfNorm(s_, einv_eff_);

  // tau-normalized versions, for the ordinary feasibility residuals (T4.1): x_hat = x/tau etc.
  const Scalar normx = normx_direct * tinv;
  const Scalar normz = normz_direct * tinv;
  const Scalar norms = norms_direct * tinv;

  m.res_dual = cinv_ * weightedInfNorm(rx_, dinv_eff_) * tinv /
               std::max(Scalar(1.0), normq_ + normx + normz);
  m.res_primal =
      weightedInfNorm(rz_, einv_eff_) * tinv / std::max(Scalar(1.0), normb_ + normx + norms);

  const Scalar xPx_tinvsq_over2 = dot_xPx_ * tinv * tinv * 0.5;
  m.cost_primal = cinv_ * (dot_qx_ * tinv + xPx_tinvsq_over2);
  m.cost_dual = cinv_ * (-dot_bz_ * tinv - xPx_tinvsq_over2);
  m.gap_abs = std::abs(m.cost_primal - m.cost_dual);
  m.gap_rel = m.gap_abs / std::max(Scalar(1.0),
                                   std::min(std::abs(m.cost_primal), std::abs(m.cost_dual)));

  m.ktratio = kappa_ * tinv;  // scale-invariant: no d/e/c weighting applies to kappa/tau at all

  // rx_inf = -A'z = rx_ + Px_ + tau*q_ (since rx_ = -(A'z + Px + tau*q));
  // rz_inf =  Ax+s = rz_ + tau*b_      (since rz_ =  Ax + s - tau*b).
  rx_inf_ = rx_ + Px_ + tau_ * q_;
  rz_inf_ = rz_ + tau_ * b_;
  // With compliance the eliminated columns of pi add C z_E (unscaled: E^-1 C_hat z_hat_E) to
  // A'z and, with pi = -z_E, to P x.
  const Scalar Cz_norm = has_compliance_ ? weightedInfNorm(Cz_, einv_eff_) : Scalar(0);
  m.res_primal_inf = std::max(cinv_ * weightedInfNorm(rx_inf_, dinv_eff_), Cz_norm) /
                     std::max(Scalar(1.0), normz_direct);
  m.res_dual_inf = std::max(
      std::max(cinv_ * weightedInfNorm(Px_, dinv_eff_), Cz_norm) / std::max(Scalar(1.0), normx_direct),
      weightedInfNorm(rz_inf_, einv_eff_) / std::max(Scalar(1.0), normx_direct + norms_direct));

  m.dot_bz = cinv_ * dot_bz_;
  m.dot_qx = cinv_ * dot_qx_;

  m.merit = std::max({m.res_primal, m.res_dual, std::min(m.gap_abs, m.gap_rel)});
  return m;
}

bool SolverImpl::isSolved(const Metrics& m, Scalar tol_feas, Scalar tol_gap_abs,
                          Scalar tol_gap_rel) const {
  const bool gap_ok = (m.gap_abs < tol_gap_abs) || (m.gap_rel < tol_gap_rel);
  return m.ktratio <= 1.0 && gap_ok && m.res_primal < tol_feas && m.res_dual < tol_feas;
}

bool SolverImpl::isPrimalInfeasible(const Metrics& m, Scalar tol_infeas_abs,
                                    Scalar tol_infeas_rel) const {
  return m.dot_bz < -tol_infeas_abs && m.res_primal_inf < -tol_infeas_rel * m.dot_bz;
}

bool SolverImpl::isDualInfeasible(const Metrics& m, Scalar tol_infeas_abs,
                                  Scalar tol_infeas_rel) const {
  return m.dot_qx < -tol_infeas_abs && m.res_dual_inf < -tol_infeas_rel * m.dot_qx;
}

void SolverImpl::updateBestIterate(const Metrics& m) {
  if (!have_best_ || m.merit < best_merit_) {
    best_x_ = x_;
    best_s_ = s_;
    best_z_ = z_;
    best_tau_ = tau_;
    best_kappa_ = kappa_;
    best_merit_ = m.merit;
    best_metrics_ = m;
    best_mu_ = computeMu();
    have_best_ = true;
  }
}

void SolverImpl::restoreBestIterate() {
  if (!have_best_) return;
  x_ = best_x_;
  s_ = best_s_;
  z_ = best_z_;
  tau_ = best_tau_;
  kappa_ = best_kappa_;
}

void SolverImpl::finalizeSolution(Status status, const Metrics& m, Scalar mu) {
  const bool primal_inf =
      (status == Status::PrimalInfeasible || status == Status::AlmostPrimalInfeasible);
  const bool dual_inf = (status == Status::DualInfeasible || status == Status::AlmostDualInfeasible);

  // Normalize infeasibility certificates so -b'z == 1 (resp. -q'x == 1) exactly, as required by
  // T4.2, rather than the ordinary 1/tau (resp. 1/kappa) scale used for an actual solution.
  Scalar scale;
  if (primal_inf) {
    scale = -1.0 / m.dot_bz;
  } else if (dual_inf) {
    scale = -1.0 / m.dot_qx;
  } else {
    scale = 1.0 / std::max(tau_, Scalar(1e-30));
  }

  sol_x_scratch_ = x_ * scale;
  sol_s_scratch_ = s_ * scale;
  sol_z_scratch_ = z_ * scale;
  if (equilibrated_) equil_.unscaleSolution(sol_x_scratch_, sol_s_scratch_, sol_z_scratch_);

  solution_.x = sol_x_scratch_;
  solution_.s = sol_s_scratch_;
  solution_.z = sol_z_scratch_;
  solution_.objective = m.cost_primal;

  solution_.info.duality_gap = m.gap_abs;
  solution_.info.primal_residual = m.res_primal;
  solution_.info.dual_residual = m.res_dual;
  solution_.info.mu = mu;
  solution_.info.kkt_refinement_residual = kkt_.lastRefinementResidual();
  solution_.info.equality_rank_deficient = kkt_.equalityRankDeficient();
  solution_.info.kkt_factorizations = kkt_.stats().factorizations;
  solution_.info.kkt_dynamic_reg_pivots = kkt_.stats().dynamic_reg_pivots;
  solution_.info.kkt_refinement_steps = kkt_.stats().refinement_steps;
  solution_.info.merit = m.merit;
  solution_.info.setup_time_s = setup_time_s_;
  solution_.info.solve_time_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start_).count();
}

Scalar SolverImpl::startFromWarmTrial() {
  cones_->zeroPrimalZeroConeBlocks(warm_s_trial_);
  cones_->projectOntoCones(warm_s_trial_);  // a user-supplied warm start need not be feasible
  cones_->projectOntoCones(warm_z_trial_);
  recenterWarmStart(warm_x_trial_, warm_s_trial_, warm_z_trial_);
  x_ = warm_x_trial_;
  s_ = warm_s_trial_;
  z_ = warm_z_trial_;
  tau_ = 1.0;
  kappa_ = warmKappa();
  computeResiduals();
  return computeMetrics().merit;
}

const Solution& SolverImpl::solve() {
  solution_.status = Status::Unsolved;
  solution_.objective = 0;
  solution_.info = Info{};
  consecutive_tiny_steps_ = 0;
  have_best_ = false;
  solve_start_ = std::chrono::steady_clock::now();
  kkt_.resetStats();
  if (!setup_done_) {
    solution_.status = Status::NumericalError;
    solution_.x.resize(0);
    solution_.s.resize(0);
    solution_.z.resize(0);
    return solution_;
  }
  kkt_.updateSettings(settings_);  // runtime settings may have changed through setSettings()
  const int verbose = settings_.verbose;
  if (verbose >= 1) {
    std::printf("conicxx: n = %d, m = %d (%d zero, %d nonneg, %d soc blocks), nnz(L) = %d\n",
                static_cast<int>(n_), static_cast<int>(m_), static_cast<int>(cone_spec_.zero_dim),
                static_cast<int>(cone_spec_.nonneg_dim),
                static_cast<int>(cone_spec_.soc_dims.size()),
                static_cast<int>(kkt_.ldlt().nonZerosL()));
  }

  // Starting point. A warm start (recentered previous solution) is used directly if its merit is
  // not above the merit of the last cold start; only otherwise is the cold start computed (one
  // factorization, two solves) and the better point kept.
  const bool use_warm = settings_.warm_start && have_warm_start_;
  Scalar warm_merit = std::numeric_limits<Scalar>::infinity();
  bool started = false;
  if (use_warm) {
    warm_x_trial_ = warm_x_;
    warm_s_trial_ = warm_s_;
    warm_z_trial_ = warm_z_;
    warm_merit = startFromWarmTrial();
    started = have_cold_merit_ && warm_merit <= cold_merit_;
  }
  if (!started) {
    if (!computeInitialPoint()) {
      solution_.status = Status::NumericalError;
      computeResiduals();
      finalizeSolution(Status::NumericalError, computeMetrics(), computeMu());
      return solution_;
    }
    computeResiduals();
    cold_merit_ = computeMetrics().merit;
    have_cold_merit_ = true;
    if (use_warm && warm_merit <= cold_merit_) {
      x_ = warm_x_trial_;
      s_ = warm_s_trial_;
      z_ = warm_z_trial_;
      tau_ = 1.0;
      kappa_ = warmKappa();
      started = true;
    }
  }

  const bool warm_used = started;
  const Scalar warm_merit_reported = use_warm ? warm_merit : Scalar(0);
  const Scalar cold_merit_reported = have_cold_merit_ ? cold_merit_ : Scalar(0);
  Status final_status = Status::MaxIterations;
  int iterations = 0;
  // Infeasibility certificates are only considered once kappa/tau exceeds this (Clarabel's gate).
  const Scalar infeas_ktratio_gate = (1.0 / settings_.tol_ktratio) * 1000.0;
  const Scalar eps = std::numeric_limits<Scalar>::epsilon();
  Metrics prev;
  Scalar alpha = 0, sigma = 0;
  if (verbose >= 2) {
    std::printf("iter     pcost          dcost         gap      pres     dres     k/t      mu       "
                "step    sigma\n");
  }

  for (int iter = 0; iter < settings_.max_iter; ++iter) {
    iterations = iter + 1;

    // Termination checks first: they need no factorization, so the final iteration costs none.
    computeResiduals();
    const Metrics m = computeMetrics();
    const Scalar mu = computeMu();
    updateBestIterate(m);
    if (verbose >= 2) {
      std::printf("%4d  %+.6e  %+.6e  %.2e %.2e %.2e %.2e %.2e  %.4f  %.4f\n", iter, m.cost_primal,
                  m.cost_dual, m.gap_abs, m.res_primal, m.res_dual, m.ktratio, mu, alpha, sigma);
    }

    if (isSolved(m, settings_.tol_feas, settings_.tol_gap_abs, settings_.tol_gap_rel)) {
      final_status = Status::Solved;
      break;
    }
    if (m.ktratio > infeas_ktratio_gate) {
      if (isPrimalInfeasible(m, settings_.tol_infeas_abs, settings_.tol_infeas_rel)) {
        final_status = Status::PrimalInfeasible;
        break;
      }
      if (isDualInfeasible(m, settings_.tol_infeas_abs, settings_.tol_infeas_rel)) {
        final_status = Status::DualInfeasible;
        break;
      }
    }
    // Progress check (Clarabel): a residual grew, and either the iterate is already accurate
    // (kappa/tau ~ 0 with a converged gap) or a residual exploded by 100x above the tolerance.
    if (iter >= 2 && (m.res_dual > prev.res_dual || m.res_primal > prev.res_primal)) {
      const bool stalled = m.ktratio < 100 * eps &&
                           (prev.gap_abs < settings_.tol_gap_abs || prev.gap_rel < settings_.tol_gap_rel);
      const bool diverging =
          m.ktratio < 1.0 &&
          ((m.res_dual > 100 * settings_.tol_feas && m.res_dual > 100 * prev.res_dual) ||
           (m.res_primal > 100 * settings_.tol_feas && m.res_primal > 100 * prev.res_primal));
      if (stalled || diverging) {
        final_status = Status::InsufficientProgress;
        break;
      }
    }
    prev = m;

    const double elapsed_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start_).count();
    if (elapsed_s > settings_.time_limit) {
      final_status = Status::MaxTime;
      break;
    }

    if (!refactorizeForCurrentScaling() || !computeConstantSolve() || !computeAffineStep()) {
      final_status = Status::NumericalError;
      break;
    }

    const Scalar alpha_aff = computeStepLength(ds_aff_, dz_aff_, dtau_aff_, dkappa_aff_);
    s_aff_scratch_ = s_ + alpha_aff * ds_aff_;
    z_aff_scratch_ = z_ + alpha_aff * dz_aff_;
    const Scalar tau_aff = tau_ + alpha_aff * dtau_aff_;
    const Scalar kappa_aff = kappa_ + alpha_aff * dkappa_aff_;
    const Scalar mu_aff = (s_aff_scratch_.dot(z_aff_scratch_) + tau_aff * kappa_aff) /
                          static_cast<Scalar>(cones_->degree() + 1);
    if (mu > 0) {
      const Scalar ratio = std::max(mu_aff / mu, Scalar(0.0));
      sigma = std::min(ratio * ratio * ratio, Scalar(1.0));
    } else {
      sigma = 1.0;
    }

    // Reduced Mehrotra correction in the first iteration (Clarabel): the starting point, cold or
    // warm, may be badly centred.
    const Scalar corrector_scale = iter == 0 ? alpha_aff : Scalar(1);
    if (!computeCombinedStep(sigma, mu, corrector_scale)) {
      final_status = Status::NumericalError;
      break;
    }

    const Scalar alpha_max =
        computeStepLength(ds_, dz_, dtau_, dkappa_) * settings_.max_step_fraction;
    if (!std::isfinite(alpha_max) || alpha_max <= 0.0) {
      final_status = Status::NumericalError;
      break;
    }
    alpha = safeguardedStepLength(alpha_max);
    if (!std::isfinite(alpha) || alpha <= 0.0) {
      final_status = Status::NumericalError;
      break;
    }
    if (alpha < settings_.min_terminate_step_length) {
      if (++consecutive_tiny_steps_ >= 2) {
        final_status = Status::InsufficientProgress;
        break;
      }
    } else {
      consecutive_tiny_steps_ = 0;
    }

    addStep(alpha);
    maybeRescale();
  }

  // MaxIterations/MaxTime/InsufficientProgress report the best iterate, upgraded to an Almost*
  // status if it meets the reduced tolerances.
  Metrics final_metrics;
  Scalar final_mu;
  if ((final_status == Status::MaxIterations || final_status == Status::MaxTime ||
       final_status == Status::InsufficientProgress) &&
      have_best_) {
    restoreBestIterate();
    final_metrics = best_metrics_;
    final_mu = best_mu_;
    if (isSolved(final_metrics, settings_.reduced_tol_feas, settings_.reduced_tol_gap_abs,
                 settings_.reduced_tol_gap_rel)) {
      final_status = Status::AlmostSolved;
    } else if (final_metrics.ktratio > (1.0 / settings_.reduced_tol_ktratio) * 1000.0) {
      if (isPrimalInfeasible(final_metrics, settings_.reduced_tol_infeas_abs,
                             settings_.reduced_tol_infeas_rel)) {
        final_status = Status::AlmostPrimalInfeasible;
      } else if (isDualInfeasible(final_metrics, settings_.reduced_tol_infeas_abs,
                                  settings_.reduced_tol_infeas_rel)) {
        final_status = Status::AlmostDualInfeasible;
      }
    }
  } else {
    // No step has been taken since the residuals were last computed.
    final_metrics = computeMetrics();
    final_mu = computeMu();
  }

  solution_.status = final_status;
  solution_.info.iterations = iterations;
  solution_.info.warm_start_used = warm_used;
  solution_.info.warm_start_merit = warm_merit_reported;
  solution_.info.cold_start_merit = cold_merit_reported;
  finalizeSolution(final_status, final_metrics, final_mu);

  if (final_status == Status::Solved && settings_.warm_start) {
    // Warm start for the next solve: x/tau, s/tau, z/tau in equilibrated units.
    const Scalar tau_inv = 1.0 / std::max(tau_, Scalar(1e-30));
    warm_x_ = x_ * tau_inv;
    warm_s_ = s_ * tau_inv;
    warm_z_ = z_ * tau_inv;
    have_warm_start_ = true;
  }
  if (verbose >= 1) {
    std::printf("conicxx: %s after %d iterations, %d factorizations, %d dynamic pivots, %.3f ms\n",
                toString(final_status), iterations, kkt_.stats().factorizations,
                kkt_.stats().dynamic_reg_pivots, 1e3 * solution_.info.solve_time_s);
  }
  return solution_;
}

}  // namespace conicxx::detail
