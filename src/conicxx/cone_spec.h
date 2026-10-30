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
  /// Exact Coulomb friction (experimental): soc_coulomb[k] != 0 makes SOC block k a Coulomb cone.
  /// Its slack s_k (the De Saxce modified velocity) then enters the primal equation as
  ///   A_k x + s_k - b_k - e_0 ||s_k,bar|| = 0,
  /// with the data already scaled to the unit cone (rows (n, mu t1, mu t2) of a contact). The
  /// problem is no longer the KKT system of an optimization problem; solve() runs the Coulomb
  /// interior-point loop (docs/design.md, "Exact Coulomb friction"). Empty: no Coulomb cones.
  std::vector<unsigned char> soc_coulomb;

  Index totalDim() const {
    Index total = zero_dim + nonneg_dim;
    for (Index d : soc_dims) total += d;
    return total;
  }

  bool hasCoulomb() const {
    for (unsigned char c : soc_coulomb) {
      if (c) return true;
    }
    return false;
  }

  bool isValid() const {
    if (zero_dim < 0 || nonneg_dim < 0) return false;
    for (Index d : soc_dims) {
      if (d < 1) return false;
    }
    if (!soc_coulomb.empty()) {
      if (soc_coulomb.size() != soc_dims.size()) return false;
      for (size_t k = 0; k < soc_dims.size(); ++k) {
        if (soc_coulomb[k] && soc_dims[k] < 2) return false;
      }
    }
    return true;
  }
};

}  // namespace conicxx
