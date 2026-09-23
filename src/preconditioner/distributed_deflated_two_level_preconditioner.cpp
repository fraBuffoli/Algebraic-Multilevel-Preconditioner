#include "distributed_deflated_two_level_preconditioner.hpp"

namespace schwarz2lvl {

void DistributedDeflatedTwoLevelPreconditioner::apply(const VectorType& r_owned, VectorType& z_owned) const {
    coarse_space_.apply(r_owned, z_coarse_);     // 1) z_0 = R_0^T A_00^{-1} R_0 r
    A_.apply(z_coarse_, A_z_coarse_);            // 2) A z_0
    r_deflated_ = r_owned - A_z_coarse_;         //    r_d = r - A z_0
    one_level_.apply(r_deflated_, z_owned);      // 3) M_RAS^{-1} r_d
    z_owned += z_coarse_;                        // 4) z = z_0 + M_RAS^{-1} r_d
}

} // namespace schwarz2lvl