#include "conicxx/sparsity_map.h"

#include <algorithm>

namespace conicxx::detail {

Index SparsityMap::addEntry(Index row, Index col) {
  entries_.emplace_back(row, col);
  return static_cast<Index>(entries_.size()) - 1;
}

SparseMat SparsityMap::finalize(Index rows, Index cols) {
  // Bucket the slots by column (counting sort), sort each bucket by row, and merge slots that
  // share a (row, col) position into one stored entry. O(nnz log(nnz per column)).
  const Index nslots = static_cast<Index>(entries_.size());
  std::vector<Index> start(static_cast<size_t>(cols) + 1, 0);
  for (const auto& e : entries_) ++start[e.second + 1];
  for (Index c = 0; c < cols; ++c) start[c + 1] += start[c];
  std::vector<Index> order(static_cast<size_t>(nslots));
  std::vector<Index> fill(start.begin(), start.end() - 1);
  for (Index s = 0; s < nslots; ++s) order[fill[entries_[s].second]++] = s;

  std::vector<Index> outer(static_cast<size_t>(cols) + 1, 0), inner;
  inner.reserve(static_cast<size_t>(nslots));
  offsets_.assign(static_cast<size_t>(nslots), -1);
  for (Index c = 0; c < cols; ++c) {
    const auto first = order.begin() + start[c], last = order.begin() + start[c + 1];
    std::sort(first, last, [this](Index a, Index b) { return entries_[a].first < entries_[b].first; });
    for (auto it = first; it != last; ++it) {
      const Index row = entries_[*it].first;
      if (static_cast<Index>(inner.size()) == outer[c] || inner.back() != row) inner.push_back(row);
      offsets_[*it] = static_cast<Index>(inner.size()) - 1;
    }
    outer[c + 1] = static_cast<Index>(inner.size());
  }
  const std::vector<Scalar> values(inner.size(), Scalar(0));
  return SparseMat(Eigen::Map<const SparseMat>(rows, cols, static_cast<Index>(inner.size()),
                                               outer.data(), inner.data(), values.data()));
}

}  // namespace conicxx::detail
