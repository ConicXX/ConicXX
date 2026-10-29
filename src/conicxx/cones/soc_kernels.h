#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include "conicxx/types.h"

/// Second-order cone kernels on raw, contiguous blocks, shared by SecondOrderCone and the batched
/// loops of ConeSet. D is the block dimension at compile time (3 for friction cones) or -1 for a
/// runtime dimension d. Formulas: see second_order_cone.h and docs/design.md.
namespace conicxx::detail::soc {

template <int D>
inline Index dimOf(Index d) {
  return D > 0 ? Index(D) : d;
}

template <int D>
inline Scalar tailNorm(const Scalar* x, Index d) {
  Scalar s = 0;
  for (Index i = 1; i < dimOf<D>(d); ++i) s += x[i] * x[i];
  return std::sqrt(s);
}

template <int D>
inline Scalar dot(const Scalar* x, const Scalar* y, Index d) {
  Scalar s = 0;
  for (Index i = 0; i < dimOf<D>(d); ++i) s += x[i] * y[i];
  return s;
}

/// out = u o v (Jordan product).
template <int D>
inline void product(const Scalar* u, const Scalar* v, Scalar* out, Index d) {
  const Index n = dimOf<D>(d);
  const Scalar u0 = u[0], v0 = v[0];
  const Scalar p0 = dot<D>(u, v, d);
  for (Index i = 1; i < n; ++i) out[i] = u0 * v[i] + v0 * u[i];
  out[0] = p0;
}

/// Solves u o out = w for out; u strictly interior. det(u) = (u0 - |u1|)(u0 + |u1|) avoids
/// cancellation near the boundary.
template <int D>
inline void inverseProduct(const Scalar* u, const Scalar* w, Scalar* out, Index d) {
  const Index n = dimOf<D>(d);
  const Scalar u0 = u[0];
  const Scalar u1n = tailNorm<D>(u, d);
  const Scalar det = (u0 - u1n) * (u0 + u1n);
  Scalar u1w1 = 0;
  for (Index i = 1; i < n; ++i) u1w1 += u[i] * w[i];
  const Scalar x0 = (u0 * w[0] - u1w1) / det;
  for (Index i = 1; i < n; ++i) out[i] = (w[i] - x0 * u[i]) / u0;
  out[0] = x0;
}

/// NT scaling point of (s, z), both strictly interior: returns eta and writes w (w'Jw = 1).
template <int D>
inline Scalar updateScaling(const Scalar* s, const Scalar* z, Scalar* w, Index d) {
  const Index n = dimOf<D>(d);
  const Scalar s1n = tailNorm<D>(s, d), z1n = tailNorm<D>(z, d);
  const Scalar a = std::sqrt((s[0] - s1n) * (s[0] + s1n));
  const Scalar b = std::sqrt((z[0] - z1n) * (z[0] + z1n));
  const Scalar gamma = std::sqrt((1.0 + dot<D>(s, z, d) / (a * b)) / 2.0);
  const Scalar inv2gamma = 1.0 / (2.0 * gamma);
  w[0] = (s[0] / a + z[0] / b) * inv2gamma;
  for (Index i = 1; i < n; ++i) w[i] = (s[i] / a - z[i] / b) * inv2gamma;
  return std::sqrt(a / b);
}

/// out = W x, W = eta [w0 w1'; w1 I + w1 w1'/(1 + w0)]. out may not alias x.
template <int D>
inline void applyW(Scalar eta, const Scalar* w, const Scalar* x, Scalar* out, Index d) {
  const Index n = dimOf<D>(d);
  Scalar w1x1 = 0;
  for (Index i = 1; i < n; ++i) w1x1 += w[i] * x[i];
  const Scalar x0 = x[0];
  const Scalar c = x0 + w1x1 / (1.0 + w[0]);
  out[0] = eta * (w[0] * x0 + w1x1);
  for (Index i = 1; i < n; ++i) out[i] = eta * (x[i] + c * w[i]);
}

/// out = W^-1 x = J W J x / eta^2. out may not alias x.
template <int D>
inline void applyWInv(Scalar eta, const Scalar* w, const Scalar* x, Scalar* out, Index d) {
  const Index n = dimOf<D>(d);
  Scalar w1x1 = 0;
  for (Index i = 1; i < n; ++i) w1x1 += w[i] * x[i];
  const Scalar x0 = x[0];
  const Scalar c = -x0 + w1x1 / (1.0 + w[0]);
  const Scalar inv = 1.0 / eta;
  out[0] = (w[0] * x0 - w1x1) * inv;
  for (Index i = 1; i < n; ++i) out[i] = (x[i] + c * w[i]) * inv;
}

/// out = H x, H = W^2 = eta^2 (2 w w' - J). out may not alias x.
template <int D>
inline void mulHs(Scalar eta, const Scalar* w, const Scalar* x, Scalar* out, Index d) {
  const Index n = dimOf<D>(d);
  const Scalar eta2 = eta * eta;
  const Scalar c = 2.0 * dot<D>(w, x, d);
  out[0] = eta2 * (c * w[0] - x[0]);
  for (Index i = 1; i < n; ++i) out[i] = eta2 * (c * w[i] + x[i]);
}

/// Lower triangle of H in row-major order (a = 0..d-1, b = 0..a).
template <int D>
inline void writeHsLowerTriangle(Scalar eta, const Scalar* w, Scalar* out, Index d) {
  const Index n = dimOf<D>(d);
  const Scalar eta2 = eta * eta;
  Index idx = 0;
  for (Index a = 0; a < n; ++a) {
    for (Index b = 0; b <= a; ++b) {
      const Scalar J = a != b ? 0.0 : (a == 0 ? 1.0 : -1.0);
      out[idx++] = eta2 * (2.0 * w[a] * w[b] - J);
    }
  }
}

template <int D>
inline Scalar margin(const Scalar* x, Index d) {
  return x[0] - tailNorm<D>(x, d);
}

/// Smallest eigenvalue of lambda o lambda for the NT point lambda of (s, z); 0 if not interior.
template <int D>
inline Scalar centrality(const Scalar* s, const Scalar* z, Index d) {
  const Scalar s1n = tailNorm<D>(s, d), z1n = tailNorm<D>(z, d);
  if (!(s[0] > s1n) || !(z[0] > z1n)) return Scalar(0);
  const Scalar dd = (s[0] - s1n) * (s[0] + s1n) * (z[0] - z1n) * (z[0] + z1n);
  const Scalar sz = dot<D>(s, z, d);
  return dd / (sz + std::sqrt(std::max(sz * sz - dd, Scalar(0))));
}

/// Euclidean projection of x onto the cone, in place.
template <int D>
inline void project(Scalar* x, Index d) {
  const Index n = dimOf<D>(d);
  const Scalar t = x[0], xn = tailNorm<D>(x, d);
  if (xn <= t) return;
  if (xn <= -t) {
    for (Index i = 0; i < n; ++i) x[i] = 0;
    return;
  }
  const Scalar a = 0.5 * (t + xn);
  x[0] = a;
  for (Index i = 1; i < n; ++i) x[i] *= a / xn;
}

/// Largest alpha in [0, alpha_max] with x + alpha dx in the cone, x strictly interior. The
/// quadratic a r^2 + b r + c with c = det(x) and a = det(dx) uses the factored determinants.
template <int D>
inline Scalar maxStep(const Scalar* x, const Scalar* dx, Scalar alpha_max, Index d) {
  // Guards an exact 0/0 only; a larger threshold misclassifies ordinary directions.
  constexpr Scalar kDegenerate = std::numeric_limits<Scalar>::min() * 1e4;
  const Index n = dimOf<D>(d);
  const Scalar p0 = x[0], d0 = dx[0];
  const Scalar p1n = tailNorm<D>(x, d), d1n = tailNorm<D>(dx, d);
  Scalar p1d1 = 0;
  for (Index i = 1; i < n; ++i) p1d1 += x[i] * dx[i];
  const Scalar a = (d0 - d1n) * (d0 + d1n);
  const Scalar b = 2.0 * (p0 * d0 - p1d1);
  const Scalar c = (p0 - p1n) * (p0 + p1n);

  const auto feasibleRoot = [&](Scalar r) { return r > 0 && (p0 + r * d0) >= 0; };
  Scalar alpha = alpha_max;
  if (std::abs(a) < kDegenerate) {
    if (std::abs(b) > kDegenerate) {
      const Scalar r = -c / b;
      if (feasibleRoot(r)) alpha = std::min(alpha, r);
    }
    return alpha;
  }
  const Scalar disc = b * b - 4.0 * a * c;
  if (disc < 0) return alpha;
  const Scalar sqrt_disc = std::sqrt(disc);
  const Scalar q = (b >= 0) ? -0.5 * (b + sqrt_disc) : -0.5 * (b - sqrt_disc);
  Scalar r1, r2;
  if (std::abs(q) > kDegenerate) {
    r1 = q / a;
    r2 = c / q;
  } else {
    r1 = r2 = -b / (2.0 * a);
  }
  if (feasibleRoot(r1)) alpha = std::min(alpha, r1);
  if (feasibleRoot(r2)) alpha = std::min(alpha, r2);
  return alpha;
}

}  // namespace conicxx::detail::soc
