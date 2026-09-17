#include "deflated_two_level_preconditioner.hpp"

namespace schwarz2lvl {

void DeflatedTwoLevelPreconditioner::apply(const VectorType& r, 
                                           VectorType& z) const {
    const auto& A = global_A_.getMatrix();
    const Eigen::Index n = A.rows();

    // Ensure output vector z is clean
    z.setZero(n);

    // Temporary vectors
    VectorType z_coarse(n);
    VectorType r_deflated(n);
    VectorType z_local(n);

    // Step 1: Compute the pure coarse space correction -> z_coarse = R_0^T * A_00^-1 * R_0 * r
    z_coarse.setZero(n);
    coarse_space_.apply(r, z_coarse);

    // Step 2: Compute the deflated residual -> r_deflated = r - A * z_coarse
    r_deflated = r - A * z_coarse;

    // Step 3: Apply the 1-level local Schwarz preconditioner to the deflated residual
    z_local.setZero(n);
    one_level_.apply(r_deflated, z_local);

    // Step 4: Final combined correction -> z = z_coarse + z_local
    z = z_coarse + z_local;
}

} // namespace schwarz2lvl
