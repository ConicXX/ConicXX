#pragma once

#include <vector>

#include "conicxx/types.h"

namespace conicxx {

/// User-facing description of the cone product K = Zero x Nonneg x SOC x ... x SOC.
///
/// Block ordering is fixed: Zero cone rows first, then the Nonnegative
/// orthant, then each second-order cone block in the order given in
/// `soc_dims`. This ordering is relied upon throughout the solver (residual
/// layout, KKT (2,2) block layout, warm-start iterate layout).
struct ConeSpec {
  Index zero_dim = 0;
  Index nonneg_dim = 0;
  std::vector<Index> soc_dims;

  Index totalDim() const {
    Index total = zero_dim + nonneg_dim;
    for (Index d : soc_dims) total += d;
    return total;
  }

  bool isValid() const {
    if (zero_dim < 0 || nonneg_dim < 0) return false;
    for (Index d : soc_dims) {
      if (d < 1) return false;
    }
    return true;
  }
};

}  // namespace conicxx
