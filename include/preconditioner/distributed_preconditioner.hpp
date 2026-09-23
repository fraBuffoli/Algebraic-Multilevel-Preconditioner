#ifndef DISTRIBUTED_PRECONDITIONER_HPP
#define DISTRIBUTED_PRECONDITIONER_HPP

#include "config.hpp"

namespace schwarz2lvl {

/**
 * @class DistributedPreconditioner
 * @brief Abstract base class for distributed preconditioners in two-level Schwarz methods.
 */
class DistributedPreconditioner {
public:
    virtual ~DistributedPreconditioner() = default;
    /**
     * @brief Applies the distributed preconditioner to the input vector.
     * @param r_owned The local vector containing the owned values (interior).
     * @param z_owned The local vector to store the result of the preconditioner application (interior).
     */
    virtual void apply(const VectorType& r_owned, VectorType& z_owned) const = 0;
};


} // namespace schwarz2lvl
#endif // DISTRIBUTED_PRECONDITIONER_HPP