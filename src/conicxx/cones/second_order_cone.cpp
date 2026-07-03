#include "conicxx/cones/second_order_cone.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace conicxx {

namespace {
// Numerical floor for quantities (rho = u0^2-||u1||^2, and u0 itself) that
// are mathematically guaranteed positive for a strictly-interior
// Nesterov-Todd point, but shrink toward zero for a near-degenerate/inactive
// cone block (e.g. a contact carrying ~zero force in a large multi-contact
// problem close to convergence). Used only in inverseProduct()/
// updateScaling() -- deliberately NOT shared with maxStep()'s own
// near-exact-zero degeneracy checks below, which are calibrated against a
// much smaller threshold and broke (mis-selecting the degenerate-quadratic
// root formula for perfectly ordinary, non-tiny step directions, verified
// via a benchmark-suite regression) the one time this was tried as a single
// shared constant. This floor is picked on the same scale as this
// codebase's other numerical-degeneracy threshold (Settings::dynamic_reg_eps,
// default 1e-14).
constexpr Scalar kTiny = 1e-12;

// Threshold for maxStep()'s quadratic-formula degeneracy checks (leading
// coefficient ~0, or the numerically-stable root q ~0): must stay near
// true zero -- it exists only to avoid an actual 0/0 -- not act as a
// "numerically dangerous" floor like kTiny above, since a and q are
// ordinary O(1)-scale quantities for typical step directions and a floor
// anywhere near kTiny's magnitude misclassifies normal (non-degenerate)
// cases as degenerate, silently picking the wrong root.
constexpr Scalar kStepDegenerateEps = std::numeric_limits<Scalar>::min() * 1e4;
}

SecondOrderCone::SecondOrderCone(Index dim)
    : ConeBase(dim),
      e_(Vec::Zero(dim)),
      W_(Mat::Identity(dim, dim)),
      Hs_(Mat::Identity(dim, dim)) {
  e_[0] = 1.0;
}

void SecondOrderCone::product(const Eigen::Ref<const Vec>& u, const Eigen::Ref<const Vec>& v,
                               Eigen::Ref<Vec> out) const {
  const Index m = dim_;
  out[0] = u.dot(v);
  out.tail(m - 1) = u[0] * v.tail(m - 1) + v[0] * u.tail(m - 1);
}

void SecondOrderCone::inverseProduct(const Eigen::Ref<const Vec>& u,
                                      const Eigen::Ref<const Vec>& w, Eigen::Ref<Vec> out) const {
  const Index m = dim_;
  Scalar u0 = u[0];
  const auto u1 = u.tail(m - 1);
  const Scalar w0 = w[0];
  const auto w1 = w.tail(m - 1);

  Scalar rho = u0 * u0 - u1.squaredNorm();
  if (std::abs(rho) < kTiny) rho = (rho >= 0 ? kTiny : -kTiny);
  // u0 is guaranteed > 0 for a valid (strictly-interior) NT point, but is
  // divided by directly below (not just via rho) -- clamp it too, or a
  // near-degenerate block poisons the result even though rho alone is fine.
  if (std::abs(u0) < kTiny) u0 = kTiny;
  const Scalar nu = u1.dot(w1);

  out[0] = (u0 * w0 - nu) / rho;
  out.tail(m - 1) = ((nu / u0 - w0) * u1 + (rho / u0) * w1) / rho;
}

void SecondOrderCone::updateScaling(const Eigen::Ref<const Vec>& s,
                                     const Eigen::Ref<const Vec>& z) {
  const Index m = dim_;
  const Scalar s0 = s[0];
  const auto s1 = s.tail(m - 1);
  const Scalar z0 = z[0];
  const auto z1 = z.tail(m - 1);

  // Floors for the margins (s0^2-||s1||^2, z0^2-||z1||^2) must scale with
  // s0^2/z0^2, not be an absolute constant: as any IPM approaches
  // convergence, complementary slackness routinely drives one side's margin
  // toward exactly zero for an active constraint -- completely normal, not
  // degenerate -- while s0/z0 itself can be any magnitude. An absolute floor
  // (verified via a benchmark-suite regression on a small, otherwise
  // perfectly ordinary RandomSOCP instance) leaves s0/a unbounded whenever
  // s0 is larger than the floor's own scale, blowing up Hs on completely
  // healthy problems. Scaling the floor by s0^2/z0^2 keeps s0/a (z0/b)
  // bounded by 1/sqrt(kTinyRel) regardless of s0/z0's absolute size; the
  // tiny absolute floor underneath only guards the literal s0==0 case.
  constexpr Scalar kTinyRel = 1e-12;
  constexpr Scalar kAbsFloor = std::numeric_limits<Scalar>::min() * 1e4;
  const Scalar a =
      std::sqrt(std::max({s0 * s0 - s1.squaredNorm(), kTinyRel * s0 * s0, kAbsFloor}));
  const Scalar b =
      std::sqrt(std::max({z0 * z0 - z1.squaredNorm(), kTinyRel * z0 * z0, kAbsFloor}));
  // beta scales W (and hence Hs=W'W ~ beta^2) directly into the KKT matrix's
  // diagonal. For a block where s and z are near the boundary by wildly
  // different amounts (e.g. one side of a contact pair pinned near-zero
  // while the other isn't -- common for a large multi-contact problem with
  // many near-inactive/near-degenerate blocks close to convergence), beta
  // grows without bound and injects an arbitrarily large entry into K,
  // blowing up its condition number until the sparse LDLT solve overflows
  // to +/-inf despite "succeeding" per Eigen::Success. Clamping beta trades
  // exactness of the NT-scaling identity (W^2 z == s) for that one
  // near-degenerate block against keeping the KKT system solvable -- the
  // same tradeoff production symmetric-cone solvers make.
  constexpr Scalar kBetaMax = 1e8;
  const Scalar beta = std::min(std::max(std::sqrt(a / b), 1.0 / kBetaMax), kBetaMax);
  const Scalar c = std::sqrt(std::max((s.dot(z) / (a * b) + 1.0) / 2.0, kTiny));

  // v = ( s/a + [z0/b; -z1/b] ) / (2c); this v satisfies the hyperbolic unit
  // norm v'Jv = 1. It is then re-normalized (v[0]+=1, v/=sqrt(2*v[0])) into
  // the Householder vector used below -- this second step is the part that
  // is easy to miss (see the class-level comment).
  Vec v(m);
  v[0] = z0 / b;
  v.tail(m - 1) = -z1 / b;
  v[0] += s0 / a;
  v.tail(m - 1) += s1 / a;
  v /= (2.0 * c);
  v[0] += 1.0;
  // v[0] is guaranteed >= 1 for s, z both strictly interior (v'Jv == 1
  // forces v[0] >= 1 before the += above even hits 0). It can dip to <= 0
  // for an iterate that has drifted a hair outside the cone by
  // floating-point round-off -- observed in practice for a near-zero-force
  // contact (s, z ~ 1e-6) close to convergence, where a fraction-to-boundary
  // step lands just past the boundary instead of just short of it. Clamp
  // rather than let sqrt() of a non-positive value hand back NaN and poison
  // the whole KKT solve for one nearly-inactive block.
  v[0] = std::max(v[0], kTiny);
  v /= std::sqrt(2.0 * v[0]);

  // W = beta * (2 v v' - J), J = diag(1, -1, ..., -1).
  W_.noalias() = 2.0 * (v * v.transpose());
  W_(0, 0) -= 1.0;
  W_.diagonal().tail(m - 1).array() += 1.0;
  W_ *= beta;

  Hs_.noalias() = W_ * W_;

  // Final safety net: the individual clamps above (rho/u0 in inverseProduct,
  // beta, v[0]) each guard one specific failure mode, but a block that's
  // degenerate in more than one of those ways at once can still compound
  // into a W/Hs that's huge-but-finite (not caught by any single clamp,
  // verified in practice: v[0]'s floor alone can still leave Hs entries
  // around 1e19+, which is finite but overflows *during* the KKT
  // factorization's internal elimination arithmetic well before any
  // per-pivot check runs, so no amount of diagonal regularization added
  // afterwards can rescue it). Falling back to an identity scaling for that
  // one block sacrifices NT-scaling exactness there in exchange for a KKT
  // system that stays solvable overall -- proportionate for a block this
  // degenerate, which is by construction carrying negligible force/slack.
  constexpr Scalar kWMax = 1e8;
  if (!W_.allFinite() || !Hs_.allFinite() || W_.cwiseAbs().maxCoeff() > kWMax) {
    W_.setIdentity();
    Hs_.setIdentity();
  }

  W_lu_.compute(W_);
}

void SecondOrderCone::applyW(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const {
  out.noalias() = W_ * x;
}

void SecondOrderCone::applyWInv(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const {
  out = W_lu_.solve(x);
}

Scalar SecondOrderCone::margin(const Eigen::Ref<const Vec>& x) const {
  const Index m = dim_;
  return x[0] - x.tail(m - 1).norm();
}

Scalar SecondOrderCone::maxStep(const Eigen::Ref<const Vec>& x, const Eigen::Ref<const Vec>& dx,
                                 Scalar alpha_max) const {
  const Index m = dim_;
  const Scalar p0 = x[0];
  const auto p1 = x.tail(m - 1);
  const Scalar d0 = dx[0];
  const auto d1 = dx.tail(m - 1);

  const Scalar a = d0 * d0 - d1.squaredNorm();
  const Scalar b = 2.0 * (p0 * d0 - p1.dot(d1));
  const Scalar c = p0 * p0 - p1.squaredNorm();  // > 0 for a strictly interior x

  auto feasibleRoot = [&](Scalar r) -> bool { return r > 0 && (p0 + r * d0) >= 0; };

  Scalar alpha = alpha_max;

  if (std::abs(a) < kStepDegenerateEps) {
    if (std::abs(b) > kStepDegenerateEps) {
      const Scalar r = -c / b;
      if (feasibleRoot(r)) alpha = std::min(alpha, r);
    }
    return alpha;
  }

  const Scalar disc = b * b - 4.0 * a * c;
  if (disc < 0) return alpha;  // never touches the boundary

  const Scalar sqrtDisc = std::sqrt(disc);
  const Scalar q = (b >= 0) ? -0.5 * (b + sqrtDisc) : -0.5 * (b - sqrtDisc);

  Scalar r1, r2;
  if (std::abs(q) > kStepDegenerateEps) {
    r1 = q / a;
    r2 = c / q;
  } else {
    r1 = r2 = -b / (2.0 * a);
  }

  if (feasibleRoot(r1)) alpha = std::min(alpha, r1);
  if (feasibleRoot(r2)) alpha = std::min(alpha, r2);
  return alpha;
}

}  // namespace conicxx
