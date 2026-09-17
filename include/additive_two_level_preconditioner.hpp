#ifndef ADDITIVE_TWO_LEVEL_PRECONDITIONER_HPP
#define ADDITIVE_TWO_LEVEL_PRECONDITIONER_HPP

#include "preconditioner.hpp"
#include "one_level_preconditioner.hpp"
#include "coarse_space.hpp"

namespace schwarz2lvl {

/**
 * @class AdditiveTwoLevelPreconditioner
 * @brief Implements the standard additive two-level Schwarz preconditioner.
 * 
 * This class combines the corrections from the 1-level subdomains and the 
 * global spectral coarse space additively: M^-1 = M_1lvl^-1 + R_0^T * A_00^-1 * R_0.
 */
class AdditiveTwoLevelPreconditioner : public Preconditioner {
public:
    /**
     * @brief Constructor coupling an existing 1-level preconditioner and a coarse space.
     * 
     * @param one_level Reference to the initialized one-level subdomain solver.
     * @param coarse_space Reference to the initialized global coarse space.
     */
    AdditiveTwoLevelPreconditioner(const OneLevelPreconditioner& one_level, 
                                   const CoarseSpace& coarse_space);

    /**
     * @brief Destructor.
     */
    virtual ~AdditiveTwoLevelPreconditioner() = default;

    /**
     * @brief Applies the additive 2-level Schwarz operator: z = (M_1lvl^-1 + M_coarse^-1) * r.
     * 
     * @param r Input global residual vector.
     * @param z Output global correction vector.
     */
    virtual void apply(const VectorType& r, 
                       VectorType& z) const override;

private:
    const OneLevelPreconditioner& one_level_; // Reference to the local 1-level preconditioner
    const CoarseSpace& coarse_space_; // Reference to the global coarse space operator
};

} // namespace schwarz2lvl

#endif // ADDITIVE_TWO_LEVEL_PRECONDITIONER_HPP
