#include "conicxx/cones/second_order_cone.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace conicxx {

namespace {
constexpr Scalar kTiny = std::numeric_limits<Scalar>::min() * 1e4;
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
  const Scalar u0 = u[0];
  const auto u1 = u.tail(m - 1);
  const Scalar w0 = w[0];
  const auto w1 = w.tail(m - 1);

  Scalar rho = u0 * u0 - u1.squaredNorm();
  if (std::abs(rho) < kTiny) rho = (rho >= 0 ? kTiny : -kTiny);
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

  const Scalar a = std::sqrt(std::max(s0 * s0 - s1.squaredNorm(), kTiny));
  const Scalar b = std::sqrt(std::max(z0 * z0 - z1.squaredNorm(), kTiny));
  const Scalar beta = std::sqrt(a / b);
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
  v /= std::sqrt(2.0 * v[0]);

  // W = beta * (2 v v' - J), J = diag(1, -1, ..., -1).
  W_.noalias() = 2.0 * (v * v.transpose());
  W_(0, 0) -= 1.0;
  W_.diagonal().tail(m - 1).array() += 1.0;
  W_ *= beta;

  Hs_.noalias() = W_ * W_;
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

  if (std::abs(a) < kTiny) {
    if (std::abs(b) > kTiny) {
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
  if (std::abs(q) > kTiny) {
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
