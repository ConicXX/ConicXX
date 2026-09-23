#pragma once

#include <chrono>
#include <memory>

#include "conicxx/cone_spec.h"
#include "conicxx/cones/cone_set.h"
#include "conicxx/kkt/equilibration.h"
#include "conicxx/kkt/kkt_system.h"
#include "conicxx/settings.h"
#include "conicxx/solution.h"
#include "conicxx/types.h"

namespace conicxx::detail {

/// Implements the homogeneous self-dual embedding interior-point method
/// described in the design notes: residuals/search-direction formulas
/// follow Domahidi/Chu/Boyd's ECOS paper (Nesterov-Todd scaling, Mehrotra
/// predictor-corrector, dynamic regularization + iterative refinement),
/// extended with the quadratic-objective (x'Px/tau) terms exactly as
/// implemented in Clarabel.rs's residuals.rs/kktsystem.rs (verified against
/// that source directly).
class SolverImpl {
 public:
  bool setup(const SparseMat& P, const Vec& q, const SparseMat& A, const Vec& b,
             const ConeSpec& cone_spec, const Settings& settings);
  bool updateData(const SparseMat* P, const Vec* q, const SparseMat* A, const Vec* b);
  void setWarmStart(const Vec& x, const Vec& s, const Vec& z);

  void setSettings(const Settings& settings) { settings_ = settings; }
  const Settings& settings() const { return settings_; }

  const Solution& solve();
  const Solution& solution() const { return solution_; }

 private:
  // --- initialization ---
  bool computeInitialPoint();
  void shiftToInteriorCold(Eigen::Ref<Vec> v) const;
  void ensureStrictlyInteriorWarm(Eigen::Ref<Vec> v) const;

  // --- per-iteration pipeline ---
  bool refactorizeForCurrentScaling();
  bool computeConstantSolve();  // x1_, z1_, Px1_
  void computeResiduals();
  Scalar computeMu() const;
  bool computeAffineStep();
  bool computeCombinedStep(Scalar sigma, Scalar mu);
  Scalar computeStepLength(const Vec& ds, const Vec& dz, Scalar dtau, Scalar dkappa) const;

  /// Backtracks alpha down from alpha_max (using ds_/dz_/dtau_/dkappa_, the combined step
  /// currently in the step-direction members) until the trial (s,z,tau,kappa) is strictly
  /// interior AND passes the centrality safeguard (Settings::centrality_theta, evaluated via
  /// lambda = W * z_trial against that same trial point's own mu), or backtracking is exhausted.
  /// T3.2 (CONICXX_AGENT_TASKS.md Phase 3): replaces the old clamp-inside-the-cone-math approach
  /// with an explicit safeguard at the one place a bad step actually gets taken.
  Scalar safeguardedStepLength(Scalar alpha_max) const;

  void addStep(Scalar alpha);
  void maybeRescale();

  Vec Pmul(const Vec& v) const;  // P (symmetric, upper-stored) * v

  // --- Phase 4 (T4.1/T4.2): termination and infeasibility, evaluated in unscaled (real-units)
  // quantities without ever reconstructing an unscaled P/A/q/b -- see docs/design.md
  // "Termination and infeasibility" for the derivation from the equilibrated internal
  // representation (x_, s_, z_, rx_, rz_, Px_, dot_qx_, dot_bz_, dot_xPx_ etc. are all in
  // equilibrated/"hat" units; every formula below algebraically undoes that, verified against a
  // genuinely-reconstructed unscaled problem in test/solver/test_termination_unscaled.cpp).

  /// Caches the always-valid (identity if !settings_.equilibrate) equilibration weight vectors
  /// and the unscaled ||b||_inf / ||q||_inf norms. Called from setup() and whenever updateData()
  /// changes b or q.
  void cacheEquilibrationWeights();
  void cacheUnscaledDataNorms(bool b_changed, bool q_changed);

  /// max_i |weights_i * v_i| (a no-op abs-max if weights is empty, i.e. equilibrate == false).
  static Scalar weightedInfNorm(const Vec& v, const Vec& weights);

  struct Metrics {
    Scalar res_primal = 0, res_dual = 0;      // unscaled, tau-normalized, T4.1
    Scalar gap_abs = 0, gap_rel = 0;          // unscaled; gap_rel uses min(|cost_primal|,|cost_dual|)
    Scalar cost_primal = 0, cost_dual = 0;    // unscaled
    Scalar ktratio = 0;                       // kappa/tau (scale-invariant, no unscaling needed)
    Scalar res_primal_inf = 0, res_dual_inf = 0;  // unscaled, NOT tau-normalized, T4.2
    Scalar dot_bz = 0, dot_qx = 0;            // unscaled, for the infeasibility certificate value
    Scalar merit = 0;                         // max(res_primal, res_dual, |gap_abs|), T4.3
  };
  /// Computes every unscaled metric from the current iterate (x_, s_, z_, tau_, kappa_) and the
  /// residual cache (computeResiduals() must have been called first this iteration).
  Metrics computeMetrics() const;

  bool isSolved(const Metrics& m, Scalar tol_feas, Scalar tol_gap_abs, Scalar tol_gap_rel) const;
  bool isPrimalInfeasible(const Metrics& m, Scalar tol_infeas_abs, Scalar tol_infeas_rel) const;
  bool isDualInfeasible(const Metrics& m, Scalar tol_infeas_abs, Scalar tol_infeas_rel) const;

  /// Snapshots (x_, s_, z_, tau_, kappa_, metrics) as the best iterate if m.merit improves on
  /// best_merit_. T4.3.
  void updateBestIterate(const Metrics& m);
  /// Restores the best snapshot into (x_, s_, z_, tau_, kappa_) -- used before finalizeSolution()
  /// on MaxIterations/MaxTime/InsufficientProgress.
  void restoreBestIterate();

  void finalizeSolution(Status status, const Metrics& m, Scalar mu);

  // --- problem data (equilibrated in place if enabled) ---
  SparseMat P_, A_;
  Vec q_, b_;
  Index n_ = 0, m_ = 0;
  ConeSpec cone_spec_;
  std::unique_ptr<ConeSet> cones_;
  Settings settings_;
  Equilibration equil_;
  KktSystem kkt_;
  bool setup_done_ = false;

  // --- iterate ---
  Vec x_, s_, z_;
  Scalar tau_ = 1.0, kappa_ = 1.0;

  // --- residual cache ---
  Vec Px_, rx_, rz_;
  Scalar rtau_ = 0, dot_qx_ = 0, dot_bz_ = 0, dot_sz_ = 0, dot_xPx_ = 0;

  // --- per-iteration constant KKT solve cache ---
  Vec x1_, z1_, Px1_;

  // --- step direction storage ---
  Vec dx_aff_, dz_aff_, ds_aff_;
  Scalar dtau_aff_ = 0, dkappa_aff_ = 0;
  Vec dx_, dz_, ds_;
  Scalar dtau_ = 0, dkappa_ = 0;

  // --- warm start ---
  bool have_warm_start_ = false;
  Vec warm_x_, warm_s_, warm_z_;

  // --- step-length safeguard state (T3.2) ---
  int consecutive_tiny_steps_ = 0;  ///< reset in solve(); see Status::InsufficientProgress

  // --- equilibration weights, always valid regardless of settings_.equilibrate (T4.1) ---
  Vec d_eff_, e_eff_, dinv_eff_, einv_eff_;  // size n, m, n, m respectively
  Scalar cinv_ = 1.0;
  Scalar normb_ = 0, normq_ = 0;  // unscaled ||b||_inf, ||q||_inf; cached, recomputed on data change

  // --- best-iterate tracking (T4.3) ---
  Vec best_x_, best_s_, best_z_;
  Scalar best_tau_ = 0, best_kappa_ = 0;
  Scalar best_merit_ = 0, best_mu_ = 0;
  Metrics best_metrics_;
  bool have_best_ = false;

  std::chrono::steady_clock::time_point solve_start_;

  Solution solution_;
};

}  // namespace conicxx::detail
