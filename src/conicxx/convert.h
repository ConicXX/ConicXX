#pragma once

#include "conicxx/types.h"

namespace conicxx {

/// Row-major sparse matrix type (CSR-like), provided purely for
/// interoperability with code that produces row-major Eigen sparse
/// matrices. ConicXX's own canonical sparse format (`SparseMat`, see
/// types.h) is column-major (CSC) -- that is the only format `Solver`
/// stores internally and reuses across `updateData()` calls.
using SparseMatRowMajor = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, Index>;

/// Converts a dense matrix to ConicXX's canonical sparse (CSC) format.
/// Every entry is kept explicit, including exact zeros: a naive conversion
/// that drops zeros (e.g. `.sparseView()`) would make the derived sparsity
/// pattern depend on which entries happen to be exactly zero on a given
/// call, silently breaking `Solver::updateData()`'s "same pattern" fast
/// path if that changes between calls. The trade-off is that a
/// dense-converted matrix carries no sparsity for the solver to exploit --
/// which is unavoidable, since a fully dense matrix has none.
///
/// Emits a one-time warning to stderr unless `warn` is false; `name` (e.g.
/// "P" or "A") is included in the message to help identify which argument
/// triggered it.
SparseMat toSparse(const Mat& dense, const char* name = nullptr, bool warn = true);

/// Like toSparse(), but keeps only the upper triangle (including the
/// diagonal), matching the storage ConicXX requires for the `P` argument
/// of `Solver::setup()`/`updateData()`. Also warns if `symmetric` is not
/// numerically symmetric (P must satisfy P = P' for the problem to be a
/// valid convex QP; ConicXX only ever reads the upper triangle, so an
/// asymmetric P would otherwise be silently misinterpreted).
SparseMat toSparseUpperTriangular(const Mat& symmetric, const char* name = nullptr,
                                  bool warn = true);

/// Converts a row-major (CSR-like) sparse matrix to ConicXX's canonical
/// column-major (CSC) format. Unlike the dense conversions above this is a
/// structure-preserving relayout (no risk to sparsity-pattern stability),
/// but it is a real one-time cost, worth flagging so a caller who repeats
/// it every timestep knows to instead construct/store the matrix in CSC
/// directly. Emits a one-time warning to stderr unless `warn` is false.
SparseMat toSparse(const SparseMatRowMajor& row_major, const char* name = nullptr,
                   bool warn = true);

}  // namespace conicxx
