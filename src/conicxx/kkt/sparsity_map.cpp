#include "conicxx/kkt/sparsity_map.h"

#include <algorithm>
#include <cassert>

namespace conicxx::detail {

Index SparsityMap::addEntry(Index row, Index col) {
  const Index slot = static_cast<Index>(entries_.size());
  entries_.emplace_back(row, col);
  return slot;
}

SparseMat SparsityMap::finalize(Index rows, Index cols) {
  // Build one triplet per UNIQUE (row, col) key -- setFromTriplets sums
  // duplicates, so if two slots aliased the same key we would otherwise not
  // be able to identify which physical entry they ended up at.
  std::map<std::pair<Index, Index>, Index> keyToTripletIndex;
  std::vector<Triplet> triplets;
  triplets.reserve(entries_.size());
  for (const auto& key : entries_) {
    if (keyToTripletIndex.find(key) == keyToTripletIndex.end()) {
      keyToTripletIndex[key] = static_cast<Index>(triplets.size());
      triplets.emplace_back(key.first, key.second, 0.0);
    }
  }

  SparseMat K(rows, cols);
  K.setFromTriplets(triplets.begin(), triplets.end());
  K.makeCompressed();

  const Index* outer = K.outerIndexPtr();
  const Index* inner = K.innerIndexPtr();

  std::map<std::pair<Index, Index>, Index> keyToOffset;
  offsets_.resize(entries_.size());
  for (size_t i = 0; i < entries_.size(); ++i) {
    const auto& key = entries_[i];
    auto it = keyToOffset.find(key);
    Index offset;
    if (it != keyToOffset.end()) {
      offset = it->second;
    } else {
      const Index col = key.second;
      const Index row = key.first;
      const Index begin = outer[col];
      const Index end = outer[col + 1];
      const Index* found = std::lower_bound(inner + begin, inner + end, row);
      assert(found != inner + end && *found == row);
      offset = static_cast<Index>(found - inner);
      keyToOffset[key] = offset;
    }
    offsets_[i] = offset;
  }
  return K;
}

}  // namespace conicxx::detail
