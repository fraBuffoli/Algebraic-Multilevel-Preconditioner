#ifndef DISTRIBUTED_DEFLATED_TWO_LEVEL_PRECONDITIONER_HPP
#define DISTRIBUTED_DEFLATED_TWO_LEVEL_PRECONDITIONER_HPP

#include "distributed_preconditioner.hpp"
#include "distributed_one_level_preconditioner.hpp"
#include "distributed_coarse_space.hpp"
#include "distributed_spmv.hpp"

namespace schwarz2lvl {

/**
 * @class DistributedDeflatedTwoLevelPreconditioner
 * @brief Distributed deflated two-level preconditioner.
 */
class DistributedDeflatedTwoLevelPreconditioner : public DistributedPreconditioner {
public:
    /**
     * @brief Constructs a DistributedDeflatedTwoLevelPreconditioner with the given distributed SpMV, one-level preconditioner, and coarse space.
     * @param A The DistributedSpMV object for applying the local matrix.
     * @param one_level The DistributedOneLevelPreconditioner object for the one-level preconditioner.
     * @param coarse_space The DistributedCoarseSpace object for the coarse space correction.
     */
    DistributedDeflatedTwoLevelPreconditioner(const DistributedSpMV& A,
                                              const DistributedOneLevelPreconditioner& one_level,
                                              const DistributedCoarseSpace& coarse_space)
        : A_(A), one_level_(one_level), coarse_space_(coarse_space) {}
    
    /**
     * @brief Applies the distributed deflated two-level preconditioner to the input vector.
     * @param r_owned The local vector containing the owned values (interior).
     * @param z_owned The local vector to store the result of the preconditioner application (interior).
     */
    void apply(const VectorType& r_owned, VectorType& z_owned) const override;

private:
    const DistributedSpMV& A_;
    const DistributedOneLevelPreconditioner& one_level_;
    const DistributedCoarseSpace& coarse_space_;
    mutable VectorType z_coarse_;
    mutable VectorType A_z_coarse_;
    mutable VectorType r_deflated_;
};

} // namespace schwarz2lvl

#endif