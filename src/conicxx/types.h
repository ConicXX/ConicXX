#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace conicxx {

using Scalar = double;
using Index = int;

using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

/// Sparse matrices use `int` as the storage index type (rather than
/// Eigen's default `Eigen::Index`/`ptrdiff_t`) to keep CSC index arrays
/// compact; this matches what most external sparsity producers (e.g.
/// contact Jacobian assemblers) use as well.
using SparseMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, Index>;
using Triplet = Eigen::Triplet<Scalar, Index>;

}  // namespace conicxx
