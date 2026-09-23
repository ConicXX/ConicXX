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
// problem close to convergence). Used only in inverseProduct() -- deliberately
// NOT shared with maxStep()'s own near-exact-zero degeneracy checks below,
// which are calibrated against a much smaller threshold and broke
// (mis-selecting the degenerate-quadratic root formula for perfectly
// ordinary, non-tiny step directions, verified via a benchmark-suite
// regression) the one time this was tried as a single shared constant. This
// floor is picked on the same scale as this codebase's other
// numerical-degeneracy threshold (Settings::regularization.dynamic_eps).
// updateScaling() itself (see T3.1/T3.2 in CONICXX_AGENT_TASKS.md) has no
// floor of its own any more -- the solver's step-length safeguard is what
// keeps s, z (and therefore u == lambda here) bounded away from the
// boundary, so this floor should be unreachable in practice, but is kept as
// a defense-in-depth guard against dividing by an exact/near-exact zero.
constexpr Scalar kTiny = 1e-12;

// Threshold for maxStep()'s quadratic-formula degeneracy checks (leading
// coefficient ~0, or the numerically-stable root q ~0): must stay near
// true zero -- it exists only to avoid an actual 0/0 -- not act as a
// "numerically dangerous" floor like kTiny above, since a and q are
// ordinary O(1)-scale quantities for typical step directions and a floor
// anywhere near kTiny's magnitude misclassifies normal (non-degenerate)
// cases as degenerate, silently picking the wrong root.
constexpr Scalar kStepDegenerateEps = std::numeric_limits<Scalar>::min() * 1e4;
}  // namespace

SecondOrderCone::SecondOrderCone(Index dim) : ConeBase(dim), e_(Vec::Zero(dim)), w_(dim), scratch_(dim) {
  e_[0] = 1.0;
  w_[0] = 1.0;
  w_.tail(dim - 1).setZero();
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
  // Closed-form arrowhead NT scaling (Domahidi/Chu/Boyd, ECOS eq. 7) -- see the class comment for
  // the derivation and the numerical verification against the previously-used Householder form.
  // No clamps or fallback here by design: s, z are required strictly interior by the caller (the
  // solver's step-length safeguard), same as this formula requires to stay well-defined.
  const Index m = dim_;
  const Scalar s0 = s[0];
  const auto s1 = s.tail(m - 1);
  const Scalar z0 = z[0];
  const auto z1 = z.tail(m - 1);

  // s'Js == (s0-||s1||)(s0+||s1||): avoids the cancellation s0^2-||s1||^2 would suffer as s
  // approaches the boundary (s0 ~ ||s1||), where the two terms are nearly equal and the naive
  // difference loses most of its significant digits.
  const Scalar s1n = s1.norm(), z1n = z1.norm();
  const Scalar a = std::sqrt((s0 - s1n) * (s0 + s1n));
  const Scalar b = std::sqrt((z0 - z1n) * (z0 + z1n));
  eta_ = std::sqrt(a / b);

  const Scalar gamma = std::sqrt((1.0 + s.dot(z) / (a * b)) / 2.0);
  const Scalar inv2gamma = 1.0 / (2.0 * gamma);
  // w = (s/a + J z/b) / (2*gamma); J flips z's tail sign.
  w_[0] = (s0 / a + z0 / b) * inv2gamma;
  w_.tail(m - 1) = (s1 / a - z1 / b) * inv2gamma;
}

void SecondOrderCone::applyW(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const {
  const Index m = dim_;
  const Scalar w0 = w_[0];
  const auto w1 = w_.tail(m - 1);
  const Scalar x0 = x[0];
  const auto x1 = x.tail(m - 1);
  const Scalar w1x1 = w1.dot(x1);
  out[0] = eta_ * (w0 * x0 + w1x1);
  out.tail(m - 1) = eta_ * (x0 * w1 + x1 + w1 * (w1x1 / (1.0 + w0)));
}

void SecondOrderCone::applyWInv(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const {
  const Index m = dim_;
  const Scalar w0 = w_[0];
  const auto w1 = w_.tail(m - 1);
  const Scalar x0 = x[0];
  const auto x1 = x.tail(m - 1);
  const Scalar w1x1 = w1.dot(x1);
  out[0] = (w0 * x0 - w1x1) / eta_;
  out.tail(m - 1) = (-x0 * w1 + x1 + w1 * (w1x1 / (1.0 + w0))) / eta_;
}

void SecondOrderCone::mulHs(const Eigen::Ref<const Vec>& x, Eigen::Ref<Vec> out) const {
  applyW(x, scratch_);
  applyW(scratch_, out);
}

void SecondOrderCone::writeHsLowerTriangle(Eigen::Ref<Vec> out) const {
  // H = W^2 = eta^2 (2 w w' - J), J = diag(1,-1,...,-1).
  const Scalar eta2 = eta_ * eta_;
  Index idx = 0;
  for (Index a = 0; a < dim_; ++a) {
    const Scalar Ja = (a == 0) ? 1.0 : -1.0;
    for (Index b = 0; b <= a; ++b) {
      const Scalar Jab = (a == b) ? Ja : 0.0;
      out[idx++] = eta2 * (2.0 * w_[a] * w_[b] - Jab);
    }
  }
}

Scalar SecondOrderCone::margin(const Eigen::Ref<const Vec>& x) const {
  const Index m = dim_;
  return x[0] - x.tail(m - 1).norm();
}

Scalar SecondOrderCone::minSquaredEigenvalue(const Eigen::Ref<const Vec>& lambda) const {
  // Eigenvalues of lambda o lambda are (lambda0 +/- ||lambda1||)^2; the "minus" one is always the
  // smaller since lambda0, ||lambda1|| >= 0 for a lambda actually produced by applyW() on an
  // interior-ish z (and the formula stays a valid, non-negative real number even if that isn't
  // quite true for some transient trial point, since it's a square either way).
  const Index m = dim_;
  const Scalar d = lambda[0] - lambda.tail(m - 1).norm();
  return d * d;
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
