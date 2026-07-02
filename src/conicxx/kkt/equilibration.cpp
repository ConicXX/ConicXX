#include "conicxx/kkt/equilibration.h"

#include <algorithm>
#include <cmath>

namespace conicxx::detail {

void Equilibration::compute(const SparseMat& P_upper, const Vec& q, const SparseMat& A,
                            const Vec& b, const ConeSet& cones, const Settings& settings) {
  (void)q;
  (void)b;
  const Index n = static_cast<Index>(P_upper.rows());
  const Index m = static_cast<Index>(A.rows());
  d_ = Vec::Ones(n);
  e_ = Vec::Ones(m);
  c_ = 1.0;

  if (!settings.equilibrate || n == 0) return;

  SparseMat Pw = P_upper;
  SparseMat Aw = A;

  const Scalar minS = settings.equilibrate_min_scale;
  const Scalar maxS = settings.equilibrate_max_scale;

  for (int iter = 0; iter < settings.equilibrate_max_iter; ++iter) {
    Vec norm_x = Vec::Zero(n);
    Vec norm_z = Vec::Zero(m);

    for (Index c = 0; c < Pw.outerSize(); ++c) {
      for (SparseMat::InnerIterator it(Pw, c); it; ++it) {
        const Index r = static_cast<Index>(it.row());
        const Scalar av = std::abs(it.value());
        norm_x[r] = std::max(norm_x[r], av);
        norm_x[c] = std::max(norm_x[c], av);
      }
    }
    for (Index c = 0; c < Aw.outerSize(); ++c) {
      for (SparseMat::InnerIterator it(Aw, c); it; ++it) {
        const Index r = static_cast<Index>(it.row());
        const Scalar av = std::abs(it.value());
        norm_x[c] = std::max(norm_x[c], av);
        norm_z[r] = std::max(norm_z[r], av);
      }
    }

    // Consolidate norm_z within each SOC block to a single shared value so
    // the resulting scaling is one scalar per block (required to preserve
    // the second-order cone's shape under the rescaling).
    for (Index bi = 0; bi < cones.numBlocks(); ++bi) {
      const ConeBase& blk = cones.block(bi);
      if (blk.type() != ConeType::SecondOrder) continue;
      const Index off = cones.blockOffset(bi);
      const Index d = blk.dim();
      const Scalar blockMax = norm_z.segment(off, d).maxCoeff();
      norm_z.segment(off, d).setConstant(blockMax);
    }

    Vec delta_x(n), delta_z(m);
    for (Index i = 0; i < n; ++i) {
      const Scalar nv = norm_x[i] > 0 ? norm_x[i] : Scalar(1.0);
      delta_x[i] = Scalar(1.0) / std::sqrt(nv);
    }
    for (Index i = 0; i < m; ++i) {
      const Scalar nv = norm_z[i] > 0 ? norm_z[i] : Scalar(1.0);
      delta_z[i] = Scalar(1.0) / std::sqrt(nv);
    }

    for (Index c = 0; c < Pw.outerSize(); ++c) {
      for (SparseMat::InnerIterator it(Pw, c); it; ++it) {
        it.valueRef() *= delta_x[it.row()] * delta_x[c];
      }
    }
    for (Index c = 0; c < Aw.outerSize(); ++c) {
      for (SparseMat::InnerIterator it(Aw, c); it; ++it) {
        it.valueRef() *= delta_z[it.row()] * delta_x[c];
      }
    }

    d_.array() *= delta_x.array();
    e_.array() *= delta_z.array();
  }

  d_ = d_.cwiseMax(minS).cwiseMin(maxS);
  e_ = e_.cwiseMax(minS).cwiseMin(maxS);

  Scalar sumAbs = 0.0;
  Index count = 0;
  for (Index c = 0; c < Pw.outerSize(); ++c) {
    for (SparseMat::InnerIterator it(Pw, c); it; ++it) {
      sumAbs += std::abs(it.value());
      ++count;
    }
  }
  if (count > 0) {
    const Scalar meanAbs = sumAbs / static_cast<Scalar>(count);
    if (meanAbs > 0) c_ = Scalar(1.0) / std::max(meanAbs, minS);
  }
  c_ = std::min(std::max(c_, minS), maxS);
}

void Equilibration::scaleP(SparseMat& P_upper) const {
  for (Index c = 0; c < P_upper.outerSize(); ++c) {
    for (SparseMat::InnerIterator it(P_upper, c); it; ++it) {
      it.valueRef() *= c_ * d_[it.row()] * d_[c];
    }
  }
}

void Equilibration::scaleA(SparseMat& A) const {
  for (Index c = 0; c < A.outerSize(); ++c) {
    for (SparseMat::InnerIterator it(A, c); it; ++it) {
      it.valueRef() *= e_[it.row()] * d_[c];
    }
  }
}

void Equilibration::scaleQ(Vec& q) const { q.array() *= c_ * d_.array(); }

void Equilibration::scaleB(Vec& b) const { b.array() *= e_.array(); }

void Equilibration::scaleProblem(SparseMat& P_upper, Vec& q, SparseMat& A, Vec& b) const {
  scaleP(P_upper);
  scaleA(A);
  scaleQ(q);
  scaleB(b);
}

void Equilibration::unscaleSolution(Vec& x, Vec& s, Vec& z) const {
  x.array() *= d_.array();
  z.array() = z.array() * e_.array() / c_;
  s.array() /= e_.array();
}

void Equilibration::scaleSolution(Vec& x, Vec& s, Vec& z) const {
  x.array() /= d_.array();
  z.array() = z.array() * c_ / e_.array();
  s.array() *= e_.array();
}

}  // namespace conicxx::detail
