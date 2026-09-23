#ifndef DISTRIBUTED_ADDITIVE_TWO_LEVEL_PRECONDITIONER_HPP
#define DISTRIBUTED_ADDITIVE_TWO_LEVEL_PRECONDITIONER_HPP

#include "distributed_preconditioner.hpp"
#include "distributed_one_level_preconditioner.hpp"
#include "distributed_coarse_space.hpp"

namespace schwarz2lvl {

/**
 * @class DistributedAdditiveTwoLevelPreconditioner
 * @brief Distributed additive two-level preconditioner.
 */
class DistributedAdditiveTwoLevelPreconditioner : public DistributedPreconditioner {
public:
    /**
     * @brief Constructs a DistributedAdditiveTwoLevelPreconditioner with the given one-level preconditioner and coarse space.
     * @param one_level The distributed one-level preconditioner.
     * @param coarse_space The distributed coarse space.
     */
    DistributedAdditiveTwoLevelPreconditioner(const DistributedOneLevelPreconditioner& one_level,
                                              const DistributedCoarseSpace& coarse_space)
        : one_level_(one_level), coarse_space_(coarse_space) {}

    /**
     * @brief Applies the distributed additive two-level preconditioner to the input vector.
     * @param r_owned The local vector containing the owned values (interior).
     * @param z_owned The local vector to store the result of the preconditioner application (interior).
     */
    void apply(const VectorType& r_owned, VectorType& z_owned) const override;

private:
    const DistributedOneLevelPreconditioner& one_level_;
    const DistributedCoarseSpace& coarse_space_;
    mutable VectorType z_coarse_;
};

} // namespace schwarz2lvl
#endif