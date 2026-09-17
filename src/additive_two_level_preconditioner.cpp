#include "additive_two_level_preconditioner.hpp"

namespace schwarz2lvl {

AdditiveTwoLevelPreconditioner::AdditiveTwoLevelPreconditioner(const OneLevelPreconditioner& one_level, 
                                                               const CoarseSpace& coarse_space)
    : one_level_(one_level), coarse_space_(coarse_space) {}

void AdditiveTwoLevelPreconditioner::apply(const VectorType& r, 
                                           VectorType& z) const {
    // Ensure the output vector z starts clean and zeroed out
    z.setZero(r.size());

    // Step 1: Apply the 1-level local Schwarz preconditioner
    // This accumulates the local subdomain corrections into z
    one_level_.apply(r, z);

    // Step 2: Apply the global Coarse Space preconditioner spectral correction
    // This adds (+=) the global low-frequency update into the same correction vector z
    coarse_space_.apply(r, z);
}

} // namespace schwarz2lvl
