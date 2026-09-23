#include "distributed_additive_two_level_preconditioner.hpp"

namespace schwarz2lvl {

void DistributedAdditiveTwoLevelPreconditioner::apply(const VectorType& r_owned, VectorType& z_owned) const {
    one_level_.apply(r_owned, z_owned);        // z   = M_RAS^{-1} r
    coarse_space_.apply(r_owned, z_coarse_);   // z_0 = R_0^T A_00^{-1} R_0 r
    z_owned += z_coarse_;
}

} // namespace schwarz2lvl