#pragma once

#include <vector>

#include "conicxx/cone_spec.h"
#include "conicxx/types.h"

namespace conicxx::detail {

/// The De Saxce term of the Coulomb cones (ConeSpec::soc_coulomb) and its derivative.
///
/// For a Coulomb cone i with slack block s_i = (s_i0, s_i,bar) the smoothed term is
///   l_i(s) = sqrt(||s_i,bar||^2 + eps^2),   L_i = dl_i/ds_i = [0, s_i,bar' / l_i],
/// eps >= 0 a smoothing velocity (0: the exact term ||s_i,bar||, L_i = 0 at s_i,bar = 0). E puts
/// one scalar per cone into the cone's first row, so the primal equation of the Coulomb problem
/// reads Ax + s - b - theta E l(s) = 0 and its derivative in s is I - theta E L. Since L_i e_0 = 0,
/// (E L)^2 = 0 and (I - theta E L)^-1 = I + theta E L =: B. All vectors here have the slack
/// dimension; non-Coulomb rows are left unchanged. docs/design.md, "Exact Coulomb friction".
class CoulombMap {
 public:
  CoulombMap() = default;
  explicit CoulombMap(const ConeSpec& spec);

  bool empty() const { return off_.empty(); }
  Index numCones() const { return static_cast<Index>(off_.size()); }
  Index offset(Index i) const { return off_[static_cast<size_t>(i)]; }
  Index dim(Index i) const { return dim_[static_cast<size_t>(i)]; }
  /// ConeSet block index of every Coulomb cone.
  const std::vector<Index>& blocks() const { return block_; }

  /// Evaluates l and L at s with smoothing eps.
  void evaluate(const Vec& s, Scalar eps);
  const Vec& ell() const { return ell_; }  ///< l_i per Coulomb cone
  Scalar eps() const { return eps_; }      ///< smoothing of the last evaluate()
  /// r_i0 -= scale * dl_i/deps = scale * eps / l_i with the evaluated l (path tangent in mu).
  void subtractEpsDerivative(Eigen::Ref<Vec> r, Scalar scale) const;
  /// The rows L_i stacked into a vector of the slack dimension (zero outside the tangential rows
  /// of the Coulomb cones), so that Hs L' is one ConeSet::mulHs().
  const Vec& jac() const { return jac_; }

  /// r_i0 -= theta l_i with the evaluated l.
  void subtractShift(Eigen::Ref<Vec> r, Scalar theta) const;
  /// r_i0 -= theta ||s_i,bar||: the unsmoothed term at s.
  void subtractExactShift(const Vec& s, Eigen::Ref<Vec> r, Scalar theta) const;
  /// v <- (I - theta E L) v = B^-1 v, in place.
  void applyBinv(Eigen::Ref<Vec> v, Scalar theta) const;
  /// v <- (I + theta E L) v = B v, in place.
  void applyB(Eigen::Ref<Vec> v, Scalar theta) const;

 private:
  std::vector<Index> off_, dim_, block_;
  Vec ell_, jac_;
  Scalar eps_ = 0;
};

}  // namespace conicxx::detail
