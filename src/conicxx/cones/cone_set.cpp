#include "conicxx/cones/cone_set.h"

#include <algorithm>
#include <limits>

#include "conicxx/cones/nonnegative_cone.h"
#include "conicxx/cones/second_order_cone.h"
#include "conicxx/cones/zero_cone.h"

namespace conicxx {

ConeSet::ConeSet(const ConeSpec& spec) {
  if (spec.zero_dim > 0) cones_.push_back(std::make_unique<ZeroCone>(spec.zero_dim));
  if (spec.nonneg_dim > 0) cones_.push_back(std::make_unique<NonnegativeCone>(spec.nonneg_dim));
  for (Index d : spec.soc_dims) cones_.push_back(std::make_unique<SecondOrderCone>(d));

  offsets_.resize(cones_.size());
  scaling_blocks_.resize(cones_.size());
  Index offset = 0;
  for (size_t i = 0; i < cones_.size(); ++i) {
    offsets_[i] = offset;
    offset += cones_[i]->dim();
    degree_ += cones_[i]->degree();
    scaling_blocks_[i] = cones_[i]->scalingBlock();
  }
  total_dim_ = offset;

  identity_.resize(total_dim_);
  for (size_t i = 0; i < cones_.size(); ++i) {
    identity_.segment(offsets_[i], cones_[i]->dim()) = cones_[i]->identityElement();
  }
}

void ConeSet::product(const Vec& u, const Vec& v, Eigen::Ref<Vec> out) const {
  for (size_t i = 0; i < cones_.size(); ++i) {
    const Index off = offsets_[i], d = cones_[i]->dim();
    cones_[i]->product(u.segment(off, d), v.segment(off, d), out.segment(off, d));
  }
}

void ConeSet::inverseProduct(const Vec& u, const Vec& w, Eigen::Ref<Vec> out) const {
  for (size_t i = 0; i < cones_.size(); ++i) {
    const Index off = offsets_[i], d = cones_[i]->dim();
    cones_[i]->inverseProduct(u.segment(off, d), w.segment(off, d), out.segment(off, d));
  }
}

void ConeSet::updateScaling(const Vec& s, const Vec& z) {
  for (size_t i = 0; i < cones_.size(); ++i) {
    const Index off = offsets_[i], d = cones_[i]->dim();
    cones_[i]->updateScaling(s.segment(off, d), z.segment(off, d));
    scaling_blocks_[i] = cones_[i]->scalingBlock();
  }
}

void ConeSet::applyW(const Vec& x, Eigen::Ref<Vec> out) const {
  for (size_t i = 0; i < cones_.size(); ++i) {
    const Index off = offsets_[i], d = cones_[i]->dim();
    cones_[i]->applyW(x.segment(off, d), out.segment(off, d));
  }
}

void ConeSet::applyWInv(const Vec& x, Eigen::Ref<Vec> out) const {
  for (size_t i = 0; i < cones_.size(); ++i) {
    const Index off = offsets_[i], d = cones_[i]->dim();
    cones_[i]->applyWInv(x.segment(off, d), out.segment(off, d));
  }
}

void ConeSet::mulHs(const Vec& x, Eigen::Ref<Vec> out) const {
  for (size_t i = 0; i < cones_.size(); ++i) {
    const Index off = offsets_[i], d = cones_[i]->dim();
    out.segment(off, d).noalias() = scaling_blocks_[i] * x.segment(off, d);
  }
}

std::pair<Scalar, Scalar> ConeSet::margins(const Vec& x) const {
  Scalar min_margin = std::numeric_limits<Scalar>::infinity();
  Scalar pos_margin_sum = 0.0;
  for (size_t i = 0; i < cones_.size(); ++i) {
    const Index off = offsets_[i], d = cones_[i]->dim();
    const Scalar m = cones_[i]->margin(x.segment(off, d));
    if (std::isfinite(m)) {
      min_margin = std::min(min_margin, m);
      pos_margin_sum += std::max(m, Scalar(0));
    }
  }
  if (!std::isfinite(min_margin)) min_margin = 0.0;  // only Zero-cone blocks present
  return {min_margin, pos_margin_sum};
}

void ConeSet::scaledUnitShift(Eigen::Ref<Vec> x, Scalar alpha) const {
  for (size_t i = 0; i < cones_.size(); ++i) {
    const Index off = offsets_[i], d = cones_[i]->dim();
    Eigen::Ref<Vec> seg = x.segment(off, d);
    cones_[i]->scaledUnitShift(seg, alpha);
  }
}

void ConeSet::zeroPrimalZeroConeBlocks(Eigen::Ref<Vec> s) const {
  for (size_t i = 0; i < cones_.size(); ++i) {
    if (cones_[i]->type() == ConeType::Zero) {
      s.segment(offsets_[i], cones_[i]->dim()).setZero();
    }
  }
}

Scalar ConeSet::maxStep(const Vec& x, const Vec& dx, Scalar alpha_max) const {
  Scalar alpha = alpha_max;
  for (size_t i = 0; i < cones_.size(); ++i) {
    const Index off = offsets_[i], d = cones_[i]->dim();
    alpha = cones_[i]->maxStep(x.segment(off, d), dx.segment(off, d), alpha);
  }
  return alpha;
}

}  // namespace conicxx
