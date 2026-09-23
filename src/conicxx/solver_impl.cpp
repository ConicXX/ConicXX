#include "conicxx/solver_impl.h"

#include <algorithm>
#include <cmath>

namespace conicxx::detail {

bool SolverImpl::setup(const SparseMat& P, const Vec& q, const SparseMat& A, const Vec& b,
                       const ConeSpec& cone_spec, const Settings& settings) {
  setup_done_ = false;
  if (!cone_spec.isValid()) return false;
  if (P.rows() != P.cols()) return false;

  n_ = static_cast<Index>(P.rows());
  m_ = cone_spec.totalDim();
  if (A.rows() != m_ || A.cols() != n_ || b.size() != m_ || q.size() != n_) return false;

  cone_spec_ = cone_spec;
  settings_ = settings;
  cones_ = std::make_unique<ConeSet>(cone_spec_);

  P_ = P;
  q_ = q;
  A_ = A;
  b_ = b;

  equil_ = Equilibration();
  if (settings_.equilibrate) {
    equil_.compute(P_, q_, A_, b_, *cones_, settings_);
    equil_.scaleProblem(P_, q_, A_, b_);
  }

  if (!kkt_.setup(P_, A_, *cones_, settings_)) return false;

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

  have_warm_start_ = false;
  solution_ = Solution{};
  setup_done_ = true;
  return true;
}

bool SolverImpl::updateData(const SparseMat* P, const Vec* q, const SparseMat* A, const Vec* b) {
  if (!setup_done_) return false;

  SparseMat P_scaled, A_scaled;
  Vec q_scaled, b_scaled;

  if (P) {
    if (P->rows() != n_ || P->cols() != n_) return false;
    P_scaled = *P;
    if (settings_.equilibrate) equil_.scaleP(P_scaled);
  }
  if (A) {
    if (A->rows() != m_ || A->cols() != n_) return false;
    A_scaled = *A;
    if (settings_.equilibrate) equil_.scaleA(A_scaled);
  }
  if (q) {
    if (q->size() != n_) return false;
    q_scaled = *q;
    if (settings_.equilibrate) equil_.scaleQ(q_scaled);
  }
  if (b) {
    if (b->size() != m_) return false;
    b_scaled = *b;
    if (settings_.equilibrate) equil_.scaleB(b_scaled);
  }

  if (!kkt_.updateData(P ? &P_scaled : nullptr, A ? &A_scaled : nullptr)) return false;

  if (P) P_ = P_scaled;
  if (A) A_ = A_scaled;
  if (q) q_ = q_scaled;
  if (b) b_ = b_scaled;

  cacheUnscaledDataNorms(b != nullptr, q != nullptr);
  return true;
}

void SolverImpl::setWarmStart(const Vec& x, const Vec& s, const Vec& z) {
  if (!setup_done_) return;
  warm_x_ = x;
  warm_s_ = s;
  warm_z_ = z;
  if (settings_.equilibrate) equil_.scaleSolution(warm_x_, warm_s_, warm_z_);
  have_warm_start_ = true;
}

Vec SolverImpl::Pmul(const Vec& v) const { return P_.selfadjointView<Eigen::Upper>() * v; }

void SolverImpl::cacheEquilibrationWeights() {
  // Always valid regardless of settings_.equilibrate: d_eff_/e_eff_ are the real Ruiz weights
  // when equilibration ran, or empty (meaning "identity", see weightedInfNorm) when it didn't --
  // every call site goes through weightedInfNorm() so this is the only place that branches on
  // settings_.equilibrate for this purpose.
  if (settings_.equilibrate) {
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

Scalar SolverImpl::weightedInfNorm(const Vec& v, const Vec& weights) {
  if (v.size() == 0) return 0.0;
  if (weights.size() == 0) return v.cwiseAbs().maxCoeff();  // equilibrate == false: identity
  return (v.array() * weights.array()).abs().maxCoeff();
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

void SolverImpl::recenterWarmStart(Eigen::Ref<Vec> s, Eigen::Ref<Vec> z) const {
  // Step 1: the smallest shift making both strictly interior, scaled to each vector's own
  // typical magnitude (pos_margin/deg) with a small coefficient -- deliberately *not*
  // shiftToInteriorCold()'s bigger target (coefficient 0.1, absolute floor 1.0): a warm point is
  // presumably already close to a good solution, so a shift sized for "no better information to
  // start from" measurably hurt it in practice (caught by test_warm_start.cpp: it made warm
  // starts need *more* iterations than cold on average for small perturbations, the opposite of
  // the point of warm-starting). Small enough to preserve the warm point's structure, but not a
  // fixed absolute epsilon either -- Phase 3 hit real trouble from exactly that (see
  // test_update_reuse.cpp), since the step-length centrality safeguard checks margin against
  // theta*mu, not an absolute constant.
  const auto shiftMinimal = [this](Eigen::Ref<Vec> v) {
    const auto [min_margin, pos_margin] = cones_->margins(v);
    const Scalar deg = static_cast<Scalar>(std::max(cones_->degree(), Index(1)));
    const Scalar target = std::max(Scalar(1e-2), Scalar(0.01) * pos_margin / deg);
    if (min_margin <= 0.0) {
      cones_->scaledUnitShift(v, -min_margin);
      cones_->scaledUnitShift(v, target);
    } else if (min_margin < target) {
      cones_->scaledUnitShift(v, target - min_margin);
    }
  };
  shiftMinimal(s);
  shiftMinimal(z);

  // Step 2: rescale (s, z) uniformly by rho so that, with tau=1 and kappa=warm_mu0 fixed,
  // mu = (s'z + kappa)/(deg+1) hits warm_mu0 exactly. Solving for rho:
  //   (rho^2 * s'z_current + warm_mu0) / (deg+1) = warm_mu0
  //   rho = sqrt(warm_mu0 * deg / s'z_current)
  const Scalar deg = static_cast<Scalar>(cones_->degree());
  const Scalar sz = s.dot(z);
  if (deg > 0.0 && sz > 0.0 && std::isfinite(sz)) {
    const Scalar rho = std::sqrt(settings_.warm_mu0 * deg / sz);
    s *= rho;
    z *= rho;
  }
}

bool SolverImpl::computeInitialPoint() {
  cones_->updateScaling(cones_->identityElement(), cones_->identityElement());
  if (!kkt_.updateScalingAndFactorize(*cones_)) return false;

  Vec rhs(n_ + m_), sol;
  if (P_.nonZeros() == 0) {
    rhs.head(n_).setZero();
    rhs.tail(m_) = b_;
    kkt_.solve(rhs, sol);
    x_ = sol.head(n_);
    s_ = -sol.tail(m_);

    rhs.head(n_) = -q_;
    rhs.tail(m_).setZero();
    kkt_.solve(rhs, sol);
    z_ = sol.tail(m_);
  } else {
    rhs.head(n_) = -q_;
    rhs.tail(m_) = b_;
    kkt_.solve(rhs, sol);
    x_ = sol.head(n_);
    z_ = sol.tail(m_);
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
  Vec rhs(n_ + m_);
  rhs.head(n_) = -q_;
  rhs.tail(m_) = b_;
  Vec sol;
  kkt_.solve(rhs, sol);
  x1_ = sol.head(n_);
  z1_ = sol.tail(m_);
  Px1_ = Pmul(x1_);
  return x1_.allFinite() && z1_.allFinite();
}

void SolverImpl::computeResiduals() {
  Px_ = Pmul(x_);
  rx_ = -(A_.transpose() * z_) - Px_ - tau_ * q_;
  rz_ = A_ * x_ + s_ - tau_ * b_;
  dot_qx_ = q_.dot(x_);
  dot_bz_ = b_.dot(z_);
  dot_sz_ = s_.dot(z_);
  dot_xPx_ = x_.dot(Px_);
  rtau_ = dot_qx_ + dot_bz_ + kappa_ + dot_xPx_ / tau_;
}

Scalar SolverImpl::computeMu() const {
  return (dot_sz_ + tau_ * kappa_) / static_cast<Scalar>(cones_->degree() + 1);
}

bool SolverImpl::computeAffineStep() {
  // Two solves feed the tau/kappa elimination: the "variable" solve (xv,zv,
  // fresh RHS below, recomputed for every affine/combined call) and the
  // "constant" solve (x1_,z1_, RHS=(-q,b), cached once per iteration by
  // computeConstantSolve()). tau_num uses the VARIABLE solve, tau_den uses
  // the CONSTANT one -- this maps onto Clarabel's kktsystem.rs exactly,
  // where (confusingly) `self.x1/z1` denotes the per-call variable solve
  // and `self.x2/z2` denotes the constant one.
  Vec rhs(n_ + m_);
  rhs.head(n_) = rx_;
  rhs.tail(m_) = s_ - rz_;
  Vec sol;
  kkt_.solve(rhs, sol);
  Vec xv = sol.head(n_), zv = sol.tail(m_);
  if (!xv.allFinite() || !zv.allFinite()) return false;

  const Scalar dtau_rhs = rtau_;
  const Scalar dkappa_rhs = tau_ * kappa_;
  const Vec xi = x_ / tau_;

  const Scalar tau_num =
      dtau_rhs - dkappa_rhs / tau_ + q_.dot(xv) + b_.dot(zv) + 2.0 * xi.dot(Pmul(xv));
  const Vec xi_minus_x1 = xi - x1_;
  const Scalar tau_den = kappa_ / tau_ - q_.dot(x1_) - b_.dot(z1_) +
                          xi_minus_x1.dot(Pmul(xi_minus_x1)) - x1_.dot(Px1_);
  if (tau_den == 0.0 || !std::isfinite(tau_num) || !std::isfinite(tau_den)) return false;

  dtau_aff_ = tau_num / tau_den;
  dx_aff_ = xv + dtau_aff_ * x1_;
  dz_aff_ = zv + dtau_aff_ * z1_;

  Vec Hz(m_);
  cones_->mulHs(dz_aff_, Hz);
  ds_aff_ = -(Hz + s_);

  dkappa_aff_ = -(dkappa_rhs + kappa_ * dtau_aff_) / tau_;

  return std::isfinite(dtau_aff_) && std::isfinite(dkappa_aff_) && dx_aff_.allFinite() &&
         dz_aff_.allFinite() && ds_aff_.allFinite();
}

bool SolverImpl::computeCombinedStep(Scalar sigma, Scalar mu) {
  const Vec dz_table = (1.0 - sigma) * rz_;

  Vec lambda(m_);
  cones_->applyW(z_, lambda);
  Vec lambda_prod(m_);
  cones_->product(lambda, lambda, lambda_prod);

  Vec Winv_ds_aff(m_);
  cones_->applyWInv(ds_aff_, Winv_ds_aff);
  Vec W_dz_aff(m_);
  cones_->applyW(dz_aff_, W_dz_aff);
  Vec corrector(m_);
  cones_->product(Winv_ds_aff, W_dz_aff, corrector);

  const Vec ds_combined = lambda_prod + corrector - sigma * mu * cones_->identityElement();

  Vec lambda_inv_ds(m_);
  cones_->inverseProduct(lambda, ds_combined, lambda_inv_ds);
  Vec ds_const(m_);
  cones_->applyW(lambda_inv_ds, ds_const);

  Vec rhs(n_ + m_);
  rhs.head(n_) = (1.0 - sigma) * rx_;
  rhs.tail(m_) = ds_const - dz_table;
  Vec sol;
  kkt_.solve(rhs, sol);
  Vec xv = sol.head(n_), zv = sol.tail(m_);
  if (!xv.allFinite() || !zv.allFinite()) return false;

  const Scalar dtau_rhs = (1.0 - sigma) * rtau_;
  const Scalar dkappa_rhs = tau_ * kappa_ + dtau_aff_ * dkappa_aff_ - sigma * mu;

  const Vec xi = x_ / tau_;
  const Scalar tau_num =
      dtau_rhs - dkappa_rhs / tau_ + q_.dot(xv) + b_.dot(zv) + 2.0 * xi.dot(Pmul(xv));
  const Vec xi_minus_x1 = xi - x1_;
  const Scalar tau_den = kappa_ / tau_ - q_.dot(x1_) - b_.dot(z1_) +
                          xi_minus_x1.dot(Pmul(xi_minus_x1)) - x1_.dot(Px1_);
  if (tau_den == 0.0 || !std::isfinite(tau_num) || !std::isfinite(tau_den)) return false;

  dtau_ = tau_num / tau_den;
  dx_ = xv + dtau_ * x1_;
  dz_ = zv + dtau_ * z1_;

  Vec Hz(m_);
  cones_->mulHs(dz_, Hz);
  ds_ = -(Hz + ds_const);

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
  Scalar alpha = alpha_max;
  Vec s_trial(m_), z_trial(m_), lambda_trial(m_);
  const Scalar deg1 = static_cast<Scalar>(cones_->degree() + 1);
  // 0.8^64 ~ 6e-7, well past min_terminate_step_length's default (1e-4) for any reasonable
  // linesearch_backtrack -- this loop always terminates via the caller's tiny-step check, not by
  // exhausting attempts on a problem that could still make progress with a smaller backtrack.
  constexpr int kMaxBacktracks = 64;
  for (int attempt = 0; attempt < kMaxBacktracks; ++attempt) {
    const Scalar tau_trial = tau_ + alpha * dtau_;
    const Scalar kappa_trial = kappa_ + alpha * dkappa_;
    if (tau_trial > 0.0 && kappa_trial > 0.0) {
      s_trial = s_ + alpha * ds_;
      z_trial = z_ + alpha * dz_;
      const auto [min_margin_s, pos_s] = cones_->margins(s_trial);
      const auto [min_margin_z, pos_z] = cones_->margins(z_trial);
      (void)pos_s;
      (void)pos_z;
      if (min_margin_s > 0.0 && min_margin_z > 0.0) {
        // Checked against mu_trial (this trial point's own mu), not the pre-step mu: a full,
        // legitimate Mehrotra step is *expected* to shrink mu substantially (that's the point of
        // taking it), so checking the resulting lambda's centrality against the old, much larger
        // mu would reject perfectly good aggressive steps, not just genuinely bad ones -- this is
        // the standard "neighborhood of the central path" formulation (N_-infinity(gamma) and
        // similar), where membership is always evaluated at the same point as the mu it's
        // compared against, not a stale one.
        const Scalar mu_trial = (s_trial.dot(z_trial) + tau_trial * kappa_trial) / deg1;
        // lambda = W * z_trial, using this iteration's already-computed NT scaling (from
        // refactorizeForCurrentScaling()'s updateScaling() call) -- cheap, and a good enough
        // proxy for the trial point's centrality without recomputing a fresh NT scaling for
        // every backtrack attempt.
        cones_->applyW(z_trial, lambda_trial);
        if (cones_->minCentrality(lambda_trial) >= settings_.centrality_theta * mu_trial) {
          return alpha;
        }
      }
    }
    alpha *= settings_.linesearch_backtrack;
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
  const Vec rx_inf = rx_ + Px_ + tau_ * q_;
  const Vec rz_inf = rz_ + tau_ * b_;
  m.res_primal_inf =
      cinv_ * weightedInfNorm(rx_inf, dinv_eff_) / std::max(Scalar(1.0), normz_direct);
  m.res_dual_inf = std::max(
      cinv_ * weightedInfNorm(Px_, dinv_eff_) / std::max(Scalar(1.0), normx_direct),
      weightedInfNorm(rz_inf, einv_eff_) / std::max(Scalar(1.0), normx_direct + norms_direct));

  m.dot_bz = cinv_ * dot_bz_;
  m.dot_qx = cinv_ * dot_qx_;

  m.merit = std::max({m.res_primal, m.res_dual, std::abs(m.gap_abs)});
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
    scale = -1.0 / (cinv_ * dot_bz_);
  } else if (dual_inf) {
    scale = -1.0 / (cinv_ * dot_qx_);
  } else {
    scale = 1.0 / std::max(tau_, Scalar(1e-30));
  }

  Vec x_out = x_ * scale, s_out = s_ * scale, z_out = z_ * scale;
  if (settings_.equilibrate) equil_.unscaleSolution(x_out, s_out, z_out);

  solution_.x = x_out;
  solution_.s = s_out;
  solution_.z = z_out;
  solution_.objective = m.cost_primal;

  solution_.info.duality_gap = m.gap_abs;
  solution_.info.primal_residual = m.res_primal;
  solution_.info.dual_residual = m.res_dual;
  solution_.info.mu = mu;
  solution_.info.kkt_refinement_residual = kkt_.lastRefinementResidual();
  solution_.info.equality_rank_deficient = kkt_.equalityRankDeficient();
  solution_.info.merit = m.merit;
}

const Solution& SolverImpl::solve() {
  solution_ = Solution{};
  consecutive_tiny_steps_ = 0;
  have_best_ = false;
  solve_start_ = std::chrono::steady_clock::now();
  if (!setup_done_) {
    solution_.status = Status::NumericalError;
    return solution_;
  }

  const bool use_warm = settings_.warm_start && have_warm_start_;
  if (use_warm) {
    // T5.2: recenter the captured warm point (shift to strictly interior, rescale to hit
    // warm_mu0), then compare its residuals against a fresh cold start and keep whichever is
    // better -- an uncentered warm start, or one carried over from a since-substantially-changed
    // problem, should never end up worse than just starting cold.
    Vec x_w = warm_x_, s_w = warm_s_, z_w = warm_z_;
    cones_->zeroPrimalZeroConeBlocks(s_w);
    recenterWarmStart(s_w, z_w);
    const Scalar tau_w = 1.0, kappa_w = settings_.warm_mu0;

    x_ = x_w;
    s_ = s_w;
    z_ = z_w;
    tau_ = tau_w;
    kappa_ = kappa_w;
    computeResiduals();
    const Scalar warm_merit = computeMetrics().merit;

    if (!computeInitialPoint()) {
      solution_.status = Status::NumericalError;
      solution_.info.iterations = 0;
      computeResiduals();
      finalizeSolution(Status::NumericalError, computeMetrics(), computeMu());
      return solution_;
    }
    computeResiduals();
    const Scalar cold_merit = computeMetrics().merit;

    if (warm_merit <= cold_merit) {
      x_ = x_w;
      s_ = s_w;
      z_ = z_w;
      tau_ = tau_w;
      kappa_ = kappa_w;
    }
    // else: keep the cold-start point computeInitialPoint() just left in x_/s_/z_/tau_/kappa_.
  } else if (!computeInitialPoint()) {
    solution_.status = Status::NumericalError;
    solution_.info.iterations = 0;
    computeResiduals();
    finalizeSolution(Status::NumericalError, computeMetrics(), computeMu());
    return solution_;
  }

  Status final_status = Status::MaxIterations;
  int iterations = 0;

  // kappa/tau must clear this before infeasibility certificates are even considered -- matches
  // Clarabel's actual gate (docs/design.md "Termination and infeasibility" explains why this
  // deviates from the task list's own plain-English paraphrase, resolved in Clarabel's favor).
  const Scalar infeas_ktratio_gate = (1.0 / settings_.tol_ktratio) * 1000.0;

  for (int iter = 0; iter < settings_.max_iter; ++iter) {
    iterations = iter + 1;

    if (!refactorizeForCurrentScaling() || !computeConstantSolve()) {
      final_status = Status::NumericalError;
      break;
    }
    computeResiduals();

    const Metrics m = computeMetrics();
    const Scalar mu = computeMu();
    updateBestIterate(m);

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

    const double elapsed_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start_).count();
    if (elapsed_s > settings_.time_limit) {
      final_status = Status::MaxTime;
      break;
    }

    if (!computeAffineStep()) {
      final_status = Status::NumericalError;
      break;
    }

    const Scalar alpha_aff = computeStepLength(ds_aff_, dz_aff_, dtau_aff_, dkappa_aff_);
    const Vec s_aff = s_ + alpha_aff * ds_aff_;
    const Vec z_aff = z_ + alpha_aff * dz_aff_;
    const Scalar tau_aff = tau_ + alpha_aff * dtau_aff_;
    const Scalar kappa_aff = kappa_ + alpha_aff * dkappa_aff_;
    const Scalar mu_aff =
        (s_aff.dot(z_aff) + tau_aff * kappa_aff) / static_cast<Scalar>(cones_->degree() + 1);

    Scalar sigma;
    if (mu > 0) {
      const Scalar ratio = std::max(mu_aff / mu, Scalar(0.0));
      sigma = std::min(std::max(ratio * ratio * ratio, Scalar(0.0)), Scalar(1.0));
    } else {
      sigma = 1.0;
    }

    if (!computeCombinedStep(sigma, mu)) {
      final_status = Status::NumericalError;
      break;
    }

    const Scalar alpha_max =
        computeStepLength(ds_, dz_, dtau_, dkappa_) * settings_.max_step_fraction;
    if (!std::isfinite(alpha_max) || alpha_max <= 0.0) {
      final_status = Status::NumericalError;
      break;
    }
    const Scalar alpha = safeguardedStepLength(alpha_max);
    if (!std::isfinite(alpha) || alpha <= 0.0) {
      final_status = Status::NumericalError;
      break;
    }

    if (alpha < settings_.min_terminate_step_length) {
      if (++consecutive_tiny_steps_ >= 2) {
        final_status = Status::InsufficientProgress;
        addStep(alpha);  // still take it -- the best available iterate is better than the last
        break;
      }
    } else {
      consecutive_tiny_steps_ = 0;
    }

    addStep(alpha);
    maybeRescale();
  }

  // For statuses that report the best iterate seen (not the last one), restore it and its
  // cached metrics now, and check whether it actually meets the reduced ("almost") tolerances --
  // if so, upgrade to the matching Almost* status (T4.3).
  Metrics final_metrics;
  Scalar final_mu;
  if (final_status == Status::MaxIterations || final_status == Status::MaxTime ||
      final_status == Status::InsufficientProgress) {
    if (have_best_) {
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
      // Shouldn't normally happen (the loop always calls updateBestIterate() before any break),
      // but fall back to whatever the current (possibly stale, e.g. pre-first-iteration) state
      // implies rather than leaving Metrics default/zero-initialized.
      final_metrics = computeMetrics();
      final_mu = computeMu();
    }
  } else {
    // Solved / PrimalInfeasible / DualInfeasible / NumericalError: x_/s_/z_/tau_/kappa_ are
    // exactly the point computeResiduals() last measured (no step taken since), so this is
    // consistent, not stale.
    final_metrics = computeMetrics();
    final_mu = computeMu();
  }

  solution_.status = final_status;
  solution_.info.iterations = iterations;
  finalizeSolution(final_status, final_metrics, final_mu);

  if (final_status == Status::Solved && settings_.warm_start) {
    // T5.1: store x/tau, s/tau, z/tau (still in equilibrated units -- x_/s_/z_/tau_ here are the
    // raw converged HSDE iterate, finalizeSolution() above only read them into local copies).
    // Storing x_/s_/z_ directly (the old bug) was only correct when tau_ happened to converge to
    // exactly 1, which it generally doesn't.
    const Scalar tau_inv = 1.0 / std::max(tau_, Scalar(1e-30));
    warm_x_ = x_ * tau_inv;
    warm_s_ = s_ * tau_inv;
    warm_z_ = z_ * tau_inv;
    have_warm_start_ = true;
  }

  return solution_;
}

}  // namespace conicxx::detail
