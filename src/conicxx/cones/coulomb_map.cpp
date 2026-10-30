#include "conicxx/cones/coulomb_map.h"

#include <cmath>

namespace conicxx::detail {

namespace {

Scalar tangentialNormSq(const Scalar* x, Index d) {
  Scalar t = 0;
  for (Index j = 1; j < d; ++j) t += x[j] * x[j];
  return t;
}

}  // namespace

CoulombMap::CoulombMap(const ConeSpec& spec) {
  Index offset = spec.zero_dim + spec.nonneg_dim;
  const Index first_block = (spec.zero_dim > 0) + (spec.nonneg_dim > 0);
  for (size_t k = 0; k < spec.soc_dims.size(); ++k) {
    if (k < spec.soc_coulomb.size() && spec.soc_coulomb[k]) {
      off_.push_back(offset);
      dim_.push_back(spec.soc_dims[k]);
      block_.push_back(first_block + static_cast<Index>(k));
    }
    offset += spec.soc_dims[k];
  }
  ell_ = Vec::Zero(numCones());
  jac_ = Vec::Zero(offset);
}

void CoulombMap::evaluate(const Vec& s, Scalar eps) {
  eps_ = eps;
  for (Index i = 0; i < numCones(); ++i) {
    const Index off = off_[i], d = dim_[i];
    const Scalar l = std::sqrt(tangentialNormSq(s.data() + off, d) + eps * eps);
    ell_[i] = l;
    const Scalar inv = l > 0 ? Scalar(1) / l : Scalar(0);
    for (Index j = 1; j < d; ++j) jac_[off + j] = s[off + j] * inv;
  }
}

void CoulombMap::subtractEpsDerivative(Eigen::Ref<Vec> r, Scalar scale) const {
  for (Index i = 0; i < numCones(); ++i) {
    if (ell_[i] > 0) r[off_[i]] -= scale * eps_ / ell_[i];
  }
}

void CoulombMap::subtractShift(Eigen::Ref<Vec> r, Scalar theta) const {
  for (Index i = 0; i < numCones(); ++i) r[off_[i]] -= theta * ell_[i];
}

void CoulombMap::subtractExactShift(const Vec& s, Eigen::Ref<Vec> r, Scalar theta) const {
  for (Index i = 0; i < numCones(); ++i) r[off_[i]] -= theta * std::sqrt(tangentialNormSq(s.data() + off_[i], dim_[i]));
}

void CoulombMap::applyBinv(Eigen::Ref<Vec> v, Scalar theta) const {
  for (Index i = 0; i < numCones(); ++i) {
    const Index off = off_[i];
    Scalar lv = 0;
    for (Index j = 1; j < dim_[i]; ++j) lv += jac_[off + j] * v[off + j];
    v[off] -= theta * lv;
  }
}

void CoulombMap::applyB(Eigen::Ref<Vec> v, Scalar theta) const {
  for (Index i = 0; i < numCones(); ++i) {
    const Index off = off_[i];
    Scalar lv = 0;
    for (Index j = 1; j < dim_[i]; ++j) lv += jac_[off + j] * v[off + j];
    v[off] += theta * lv;
  }
}

}  // namespace conicxx::detail
