#pragma once

#include "conicxx/cones/cone_set.h"
#include "conicxx/settings.h"
#include "conicxx/types.h"

namespace conicxx::detail {

/// Ruiz-style diagonal equilibration of the problem data, computed once at
/// setup() time to improve KKT conditioning for badly-scaled problems (e.g.
/// physical units mixed across force/velocity/impulse scales in a multibody
/// contact problem).
///
/// The equilibrated problem (in variables x_hat, s_hat, z_hat) is
///   min  0.5 x_hat' P_hat x_hat + q_hat' x_hat
///   s.t. A_hat x_hat + s_hat = b_hat,  s_hat in K
/// with
///   P_hat = c * D * P * D,   q_hat = c * D * q
///   A_hat = E * A * D,       b_hat = E * b
/// for positive diagonal D (size n), E (size m, constant within each
/// second-order-cone block so the cone's shape is preserved), and scalar
/// c > 0. The original solution is recovered via
///   x = D * x_hat,   s = E^-1 * s_hat,   z = (1/c) * E * z_hat.
class Equilibration {
 public:
  /// Compute D, E, c from the given (unscaled) problem data via a fixed
  /// number of Ruiz iterations, without modifying P_upper/A themselves.
  void compute(const SparseMat& P_upper, const Vec& q, const SparseMat& A, const Vec& b,
               const ConeSet& cones, const Settings& settings);

  /// Apply the (already-computed) scaling to problem data in place. Each
  /// piece can also be scaled independently -- useful when only some of
  /// (P, q, A, b) change between timesteps.
  void scaleProblem(SparseMat& P_upper, Vec& q, SparseMat& A, Vec& b) const;
  void scaleP(SparseMat& P_upper) const;
  void scaleA(SparseMat& A) const;
  void scaleQ(Vec& q) const;
  void scaleB(Vec& b) const;

  void unscaleSolution(Vec& x, Vec& s, Vec& z) const;
  /// Inverse of unscaleSolution: map an original-units (x, s, z) into the
  /// equilibrated-space representation (used to ingest a user-supplied warm
  /// start, which is expressed in original units).
  void scaleSolution(Vec& x, Vec& s, Vec& z) const;
  Scalar unscaleObjective(Scalar scaled_objective) const { return scaled_objective / c_; }

  const Vec& d() const { return d_; }
  const Vec& e() const { return e_; }
  Scalar c() const { return c_; }

 private:
  Vec d_;  // size n
  Vec e_;  // size m
  Scalar c_ = 1.0;
};

}  // namespace conicxx::detail
