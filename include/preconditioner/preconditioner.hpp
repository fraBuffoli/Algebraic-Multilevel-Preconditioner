/**
 * @file preconditioner.hpp
 * @brief Abstract preconditioner interface M^{-1} used by the Krylov solver.
 */
#ifndef SCHWARZ2LVL_PRECONDITIONER_PRECONDITIONER_HPP
#define SCHWARZ2LVL_PRECONDITIONER_PRECONDITIONER_HPP

#include "types.hpp"

#include <string>

namespace schwarz2lvl::prec {

/**
 * @class Preconditioner
 * @brief Application of a (right) preconditioner z = M^{-1} r.
 *
 * Vectors are distributed: @p r and @p z contain the owned entries only.
 * apply() is collective over the communicator of the underlying operator.
 */
class Preconditioner {
public:
    virtual ~Preconditioner() = default;

    /**
     * @brief z = M^{-1} r.
     * @param r Owned entries of the input vector.
     * @param z Owned entries of the output vector (resized).
     */
    virtual void apply(const core::Vec& r, core::Vec& z) const = 0;

    /// @brief Short human readable name.
    virtual std::string name() const = 0;
};

/**
 * @class IdentityPreconditioner
 * @brief M = I (unpreconditioned GMRES).
*/
class IdentityPreconditioner final : public Preconditioner {
public:
    void apply(const core::Vec& r, core::Vec& z) const override { z = r; }
    std::string name() const override { return "none"; }
};

} // namespace schwarz2lvl::prec

#endif // SCHWARZ2LVL_PRECONDITIONER_PRECONDITIONER_HPP
