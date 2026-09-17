#ifndef DEFLATED_TWO_LEVEL_PRECONDITIONER_HPP
#define DEFLATED_TWO_LEVEL_PRECONDITIONER_HPP

#include "preconditioner.hpp"
#include "one_level_preconditioner.hpp"
#include "coarse_space.hpp"

namespace schwarz2lvl {

/**
 * @class DeflatedTwoLevelPreconditioner
 * @brief Implements the advanced deflated two-level Schwarz preconditioner.
 * 
 * This class projects out the low-frequency error components from the residual 
 * BEFORE applying the 1-level local solver, accelerating Krylov subspace convergence.
 */
class DeflatedTwoLevelPreconditioner : public Preconditioner {
public:
    /**
     * @brief Constructor coupling the global matrix, 1-level solver, and coarse space.
     * 
     * @param global_A Reference to the global sparse matrix A (needed for residual correction).
     * @param one_level Reference to the initialized one-level subdomain solver.
     * @param coarse_space Reference to the initialized global coarse space.
     */
    DeflatedTwoLevelPreconditioner(const SparseMatrixWrapper& global_A,
                                   const OneLevelPreconditioner& one_level, 
                                   const CoarseSpace& coarse_space)
                                   : global_A_(global_A), one_level_(one_level), coarse_space_(coarse_space) {};

    /**
     * @brief Destructor.
     */
    virtual ~DeflatedTwoLevelPreconditioner() = default;

    /**
     * @brief Applies the deflated 2-level Schwarz operator.
     * 
     * @param r Input global residual vector.
     * @param z Output global correction vector.
     */
    virtual void apply(const VectorType& r, 
                       VectorType& z) const override;

private:
    const SparseMatrixWrapper& global_A_; // Reference to the global sparse matrix
    const OneLevelPreconditioner& one_level_; // Reference to the local 1-level preconditioner
    const CoarseSpace& coarse_space_; // Reference to the global coarse space operator
};

} // namespace schwarz2lvl

#endif // DEFLATED_TWO_LEVEL_PRECONDITIONER_HPP
