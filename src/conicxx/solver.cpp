#include "conicxx/solver.h"

#include "conicxx/solver_impl.h"

namespace conicxx {

Solver::Solver() : impl_(std::make_unique<detail::SolverImpl>()) {}
Solver::~Solver() = default;
Solver::Solver(Solver&&) noexcept = default;
Solver& Solver::operator=(Solver&&) noexcept = default;

bool Solver::setup(const SparseMat& P, const Vec& q, const SparseMat& A, const Vec& b,
                   const ConeSpec& cone_spec, const Settings& settings) {
  return impl_->setup(P, q, A, b, cone_spec, settings);
}

bool Solver::setup(const Mat& P, const Vec& q, const SparseMat& A, const Vec& b,
                   const ConeSpec& cone_spec, const Settings& settings) {
  return setup(toSparseUpperTriangular(P, "P"), q, A, b, cone_spec, settings);
}

bool Solver::setup(const SparseMat& P, const Vec& q, const Mat& A, const Vec& b,
                   const ConeSpec& cone_spec, const Settings& settings) {
  return setup(P, q, toSparse(A, "A"), b, cone_spec, settings);
}

bool Solver::setup(const Mat& P, const Vec& q, const Mat& A, const Vec& b,
                   const ConeSpec& cone_spec, const Settings& settings) {
  return setup(toSparseUpperTriangular(P, "P"), q, toSparse(A, "A"), b, cone_spec, settings);
}

bool Solver::updateData(const SparseMat* P, const Vec* q, const SparseMat* A, const Vec* b) {
  return impl_->updateData(P, q, A, b);
}

void Solver::setWarmStart(const Vec& x, const Vec& s, const Vec& z) {
  impl_->setWarmStart(x, s, z);
}

void Solver::setSettings(const Settings& settings) { impl_->setSettings(settings); }
const Settings& Solver::settings() const { return impl_->settings(); }

const Solution& Solver::solve() { return impl_->solve(); }
const Solution& Solver::solution() const { return impl_->solution(); }

}  // namespace conicxx
