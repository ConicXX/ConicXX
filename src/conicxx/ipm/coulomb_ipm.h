#pragma once

#include <chrono>
#include <vector>

#include "conicxx/cone_spec.h"
#include "conicxx/cones/cone_set.h"
#include "conicxx/cones/coulomb_map.h"
#include "conicxx/kkt/kkt_system.h"
#include "conicxx/settings.h"
#include "conicxx/solution.h"
#include "conicxx/types.h"

namespace conicxx::detail {

/// The data of the (equilibrated) problem as SolverImpl holds them, and the weights that map
/// residuals back to user units (empty weight vectors = identity), see SolverImpl::computeMetrics().
struct CoulombProblem {
  const SparseMat& P;  ///< upper triangle
  const Vec& q;
  const SparseMat& A;
  const Vec& b;
  const Vec& compliance;  ///< zero-cone compliance (size zero_dim)
  bool has_compliance;
  Index zero_dim;
  ConeSet& cones;
  KktSystem& kkt;
  const Vec& d;
  const Vec& e;
  const Vec& dinv;
  const Vec& einv;
  Scalar cinv;
  Scalar normb, normq;  ///< unscaled ||b||_inf, ||q||_inf
};

/// Primal-dual interior-point method for the exact Coulomb friction problem
///
///   Px + q + A'z = 0,   Ax + s - b - C z_E - E l(s) = 0,   s o z = 0,  s, z in K,
///
/// with l the De Saxce term of the Coulomb cones (CoulombMap). Not an optimization problem, so
/// there is no homogeneous embedding: an infeasible-start Mehrotra predictor-corrector on the
/// perturbed system s o z = mu e with Nesterov-Todd scaling, the De Saxce term smoothed with
/// eps = c mu (CoulombSettings::smoothing). The Newton matrix is the convex KKT matrix with the
/// (2,2) block of every Coulomb cone replaced by -(I - E L) Hs, factorized by an unsymmetric LDU on
/// the symbolic structure of the convex one (KktSystem). docs/design.md, "Exact Coulomb friction".
class CoulombIpm {
 public:
  struct Metrics {
    Scalar res_primal = 0, res_dual = 0;  ///< unscaled, relative, unsmoothed De Saxce term
    /// max_i v_i / (tol_gap_abs + tol_gap_rel ||s_i||) with v_i = ||s_i o z_i|| / (||z_i|| +
    /// m_i ||s_i||), m_i the effective mass of cone i: the velocity error of an active and the
    /// velocity induced by the residual impulse of a separating contact, relative to the tolerance
    /// (<= 1: converged), unscaled.
    Scalar comp = 0;
    Scalar comp_abs = 0;    ///< max_i ||s_i o z_i||, unscaled (reported as the duality gap)
    Scalar comp_bound = 0;  ///< min_i of the ||s_i o z_i|| that meets the tolerance, scaled units
    Scalar merit = 0;       ///< max(res_primal / tol_feas, res_dual / tol_feas, comp)
  };
  struct Result {
    Status status = Status::Unsolved;
    int iterations = 0;
    int continuation_steps = 0;  ///< accepted pseudo-arclength steps
    Metrics metrics;
    Scalar mu = 0;
    Scalar smoothing = 0;
  };

  explicit CoulombIpm(const ConeSpec& spec, Index n, Index m);

  const CoulombMap& map() const { return map_; }

  /// Metrics of (x, s, z) in equilibrated units.
  Metrics metrics(const CoulombProblem& pb, const Settings& settings, const Vec& x, const Vec& s, const Vec& z);

  /// Runs the method from (x, s, z) (equilibrated units, s and z strictly interior) and leaves the
  /// reported iterate (the best one unless Solved) in (x, s, z).
  Result solve(const CoulombProblem& pb, const Settings& settings, Vec& x, Vec& s, Vec& z,
               std::chrono::steady_clock::time_point start);

 private:
  template <typename Derived>
  void Pmul(const CoulombProblem& pb, const Eigen::MatrixBase<Derived>& v, Eigen::Ref<Vec> out) const {
    out.noalias() = pb.P.selfadjointView<Eigen::Upper>() * v;
  }
  /// rp = Ax + s - b - C z_E (without the De Saxce term), rd = Px + q + A'z.
  void linearResiduals(const CoulombProblem& pb, const Vec& x, const Vec& s, const Vec& z);
  bool solveNewton(const CoulombProblem& pb, const Vec& ds_const, Vec& dx, Vec& dz, Vec& ds);
  Scalar maxStep(const CoulombProblem& pb, const Vec& s, const Vec& z, const Vec& ds, const Vec& dz) const;
  /// Linearization at (x, s, z) with the smoothing eps: residuals rd_ and rp_ (smoothed), NT
  /// scaling and factorization of the Newton matrix. The linear residuals are recomputed unless
  /// have_residuals (metrics() computed them at this point).
  bool linearize(const CoulombProblem& pb, const Vec& x, const Vec& s, const Vec& z, Scalar eps,
                 bool have_residuals);
  /// ds_const of the pure centering (Newton) step towards s o z = tau e: V (v \ (v o v - tau e)).
  void centeringConst(const CoulombProblem& pb, const Vec& z, Scalar tau);
  /// Tangent dy/dmu of the central path at the linearized point into tx_, ts_, tz_ (c: smoothing
  /// scale, eps = c mu). Uses the factorization of linearize().
  bool tangent(const CoulombProblem& pb, const Vec& z, Scalar c);
  enum class PathResult { HandBack, Failed };
  /// Pseudo-arclength continuation of the central path from the stalled iterate (x, s, z), where
  /// mu cannot be decreased (a fold of the path in mu, docs/design.md): predictor along the tangent,
  /// bordered Newton corrector with mu free, in a local relative metric. Oriented by the last
  /// Mehrotra direction (dx_, ds_, dz_). Returns HandBack once mu is below mu_exit on the path.
  PathResult continuePath(const CoulombProblem& pb, const Settings& settings, Vec& x, Vec& s, Vec& z,
                          Scalar c, Scalar nu, Scalar mu_exit, int& steps);
  /// Weighted inner product of the continuation metric (weights wx_, ws_, wz_, wtau_).
  Scalar pathDot(const Vec& ax, const Vec& as, const Vec& az, Scalar at, const Vec& bx, const Vec& bs,
                 const Vec& bz, Scalar bt) const;
  void pathWeights(const CoulombProblem& pb, const Vec& x, const Vec& s, const Vec& z, Scalar tau);

  /// Verbose diagnostics of the central path at the current iterate (factorization and scaling of
  /// this iteration): the tangent dy/dmu from one extra solve, reported per cone as the local
  /// orders mu ||ds_i/dmu|| / ||s_i|| and mu ||dz_i/dmu|| / ||z_i|| (0: component stays O(1),
  /// 1: vanishes like mu, 1/2: degenerate; >> 1: nearly singular Jacobian, a turning point), and
  /// the number of pivots of unexpected sign (a change of its parity is a sign change of det J).
  void printPathMonitor(const CoulombProblem& pb, const Vec& s, const Vec& z, Scalar mu, Scalar c);
  Scalar safeguardedStep(const CoulombProblem& pb, const Settings& settings, const Vec& s, const Vec& z,
                         Scalar alpha_max);

  CoulombMap map_;
  Index n_ = 0, m_ = 0;
  Vec mob_;  // (A P^-1 A')_rr with the diagonal of P, scaled, per slack row
  Vec Px_, Atz_, Ax_, rd_, rp_, rhs_, sol_, tmp_m_, row0_, prod_;
  Vec dx_aff_, dz_aff_, ds_aff_, dx_, dz_, ds_;
  Vec lambda_, lambda_prod_, Winv_ds_aff_, W_dz_aff_, corrector_, ds_comb_, lambda_inv_ds_, ds_const_, Hz_;
  Vec s_trial_, z_trial_;
  Vec tx_, tz_, ts_, tmp2_m_;  // path tangent
  // continuation: unit tangent, predicted point, corrector point and updates, metric weights
  Vec ux_, us_, uz_, px_, ps_, pz_, cx_, cs_, cz_, ax_, as_, az_;
  Vec ws_, wz_;
  Scalar wx_ = 1, wtau_ = 1;
  // Last iterate with centrality >= kCenteredForPath * mu and its direction: the continuation
  // starts there (the stalled iterate itself is usually far from the path).
  Vec gx_, gs_, gz_, gdx_, gds_, gdz_;
  bool have_centered_ = false;
  Vec best_x_, best_s_, best_z_;
};

/// Classifies every Coulomb cone of map by the spectral values lambda_1 <= lambda_2 of s_i and z_i
/// in equilibrated units, where s and z have comparable scales: a spectral value counts as zero
/// below threshold. Near the central path with mu << 1 the vanishing ones are O(mu) and the others
/// O(1), so threshold = sqrt(mu) separates them.
std::vector<ContactState> classifyContacts(const CoulombMap& map, const Vec& s, const Vec& z, Scalar threshold);

}  // namespace conicxx::detail
