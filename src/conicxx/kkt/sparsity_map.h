#pragma once

#include <map>
#include <utility>
#include <vector>

#include "conicxx/types.h"

namespace conicxx::detail {

/// Builds a sparse matrix once from a set of registered (row, col) logical
/// entries ("slots"), then lets the caller overwrite any slot's numeric
/// value directly in the matrix's CSC value array -- no re-triangulation,
/// no repeated `setFromTriplets`. This is the mechanism that lets KktSystem
/// call `analyzePattern()` exactly once and reuse it both across IPM
/// iterations (only the Hs scaling block changes) and across solver
/// timesteps (only P/A numeric values change, when the sparsity pattern is
/// unchanged).
///
/// Multiple slots may be registered at the same (row, col); the underlying
/// matrix only has one physical nonzero there; each slot still resolves
/// independently to that shared offset, so writing through any of them
/// updates the same entry (this is only used for genuinely-shared physical
/// entries such as a diagonal that could receive contributions from more
/// than one logical source -- callers otherwise register at most one slot
/// per position to avoid surprising aliasing).
class SparsityMap {
 public:
  /// Register a logical entry at (row, col). Returns a slot id, stable
  /// after finalize().
  Index addEntry(Index row, Index col);

  /// Build the sparse matrix (all entries initialized to 0) and resolve
  /// every registered slot to its final position in the CSC value array.
  SparseMat finalize(Index rows, Index cols);

  Index numSlots() const { return static_cast<Index>(offsets_.size()); }

  void setValue(SparseMat& K, Index slot, Scalar value) const {
    K.valuePtr()[offsets_[static_cast<size_t>(slot)]] = value;
  }

  Scalar getValue(const SparseMat& K, Index slot) const {
    return K.valuePtr()[offsets_[static_cast<size_t>(slot)]];
  }

 private:
  std::vector<std::pair<Index, Index>> entries_;  // slot -> (row, col), registration order
  std::vector<Index> offsets_;                    // slot -> CSC value-array offset
};

}  // namespace conicxx::detail
