#include "conicxx/convert.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace conicxx {

namespace {

void warnOnce(const char* message, const char* name, bool warn) {
  if (!warn) return;
  if (name && *name) {
    std::fprintf(stderr, "[conicxx] warning: %s (matrix: %s)\n", message, name);
  } else {
    std::fprintf(stderr, "[conicxx] warning: %s\n", message);
  }
}

}  // namespace

SparseMat toSparse(const Mat& dense, const char* name, bool warn) {
  warnOnce(
      "dense matrix converted to sparse -- every entry (including exact zeros) is kept explicit "
      "to keep the sparsity pattern stable across repeated conversions, so this forgoes sparsity "
      "exploitation entirely; construct a conicxx::SparseMat directly for large or genuinely "
      "sparse problems",
      name, warn);

  std::vector<Triplet> triplets;
  triplets.reserve(static_cast<size_t>(dense.rows()) * static_cast<size_t>(dense.cols()));
  for (Index j = 0; j < dense.cols(); ++j) {
    for (Index i = 0; i < dense.rows(); ++i) {
      triplets.emplace_back(i, j, dense(i, j));
    }
  }
  SparseMat out(dense.rows(), dense.cols());
  out.setFromTriplets(triplets.begin(), triplets.end());
  return out;
}

SparseMat toSparseUpperTriangular(const Mat& symmetric, const char* name, bool warn) {
  if (warn && symmetric.rows() == symmetric.cols() && symmetric.rows() > 0) {
    const Scalar asym = (symmetric - symmetric.transpose()).cwiseAbs().maxCoeff();
    const Scalar scale = std::max(Scalar(1.0), symmetric.cwiseAbs().maxCoeff());
    if (asym > 1e-8 * scale) {
      warnOnce(
          "matrix passed as P is not numerically symmetric -- ConicXX only reads the upper "
          "triangle, so the lower triangle is silently ignored; verify P == P'",
          name, warn);
    }
  }
  warnOnce(
      "dense matrix converted to sparse (upper triangle only) for use as P -- forgoes sparsity "
      "exploitation; construct a conicxx::SparseMat directly for large or genuinely sparse "
      "problems",
      name, warn);

  std::vector<Triplet> triplets;
  const Index n = symmetric.rows();
  triplets.reserve(static_cast<size_t>(n) * static_cast<size_t>(n + 1) / 2);
  for (Index j = 0; j < symmetric.cols(); ++j) {
    for (Index i = 0; i <= j && i < symmetric.rows(); ++i) {
      triplets.emplace_back(i, j, symmetric(i, j));
    }
  }
  SparseMat out(symmetric.rows(), symmetric.cols());
  out.setFromTriplets(triplets.begin(), triplets.end());
  return out;
}

SparseMat toSparse(const SparseMatRowMajor& row_major, const char* name, bool warn) {
  warnOnce(
      "row-major (CSR) sparse matrix converted to ConicXX's canonical column-major (CSC) layout "
      "-- construct the matrix directly in CSC (conicxx::SparseMat) to avoid this one-time "
      "relayout cost",
      name, warn);
  return SparseMat(row_major);
}

}  // namespace conicxx
