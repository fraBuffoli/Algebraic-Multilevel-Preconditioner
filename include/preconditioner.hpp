#ifndef PRECONDITIONER_HPP
#define PRECONDITIONER_HPP

#include "sparse_matrix.hpp"

namespace schwarz2lvl {

/**
 * @class Preconditioner
 * @brief Abstract base class defining the interface for all preconditioners.
 * 
 * Any iterative linear solver (like GMRES) will interact with preconditioners 
 * exclusively through this interface by calling the virtual apply() method.
 */
class Preconditioner {
public:
    /**
     * @brief Virtual destructor to ensure proper memory cleanup of derived classes.
     */
    virtual ~Preconditioner() = default;

    /**
     * @brief Applies the preconditioner operator to a given vector: z = M^-1 * r.
     * 
     * This method must be overridden by any specific preconditioner implementation 
     * (e.g., OneLevelPreconditioner, TwoLevelPreconditioner).
     * 
     * @param r Input vector (typically the current residual from the Krylov solver).
     * @param z Output vector where the computed preconditioned correction is stored.
     */
    virtual void apply(const VectorType& r, VectorType& z) const = 0;
};

} // namespace schwarz2lvl

#endif // PRECONDITIONER_HPP
