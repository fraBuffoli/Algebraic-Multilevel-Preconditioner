/**
 * @file preconditioner.hpp
 * @brief Abstract preconditioner interface M^{-1} used by the Krylov solver.
 */
#ifndef SCHWARZ2LVL_PRECONDITIONER_PRECONDITIONER_HPP
#define SCHWARZ2LVL_PRECONDITIONER_PRECONDITIONER_HPP

#include "types.hpp"

#include <mpi.h>

#include <string>

namespace schwarz2lvl::prec {

/**
 * @brief How much of the setup is redone when the matrix values change (same sparsity pattern).
 *
 * In a CFD time loop the pattern of the pressure matrix never changes, the values change a
 * little at every time step and not at all between the correctors of one step. The setup is
 * therefore split by what each part depends on:
 *  - Reuse:       nothing is recomputed (the matrix did not change, or a lagged preconditioner is wanted);
 *  - Refactorize: numerical LU of A_ii (symbolic analysis kept), A_00 = R_0 A R_0^T with the *old*
 *                 coarse basis and numerical LU of A_00. Same idea as GAMG, which keeps its
 *                 agglomeration and only recomputes the coarse matrices;
 *  - Rebuild:     everything that depends on the values, including the local eigenproblems and
 *                 the coarse basis (the subdomain structure, i.e. overlap and halo, is kept).
 * The numeric values are ordered: a larger level contains the smaller ones.
 */
enum class UpdateLevel : int { Reuse = 0, Refactorize = 1, Rebuild = 2 };

/// @brief Short name of an update level ("reuse", "refactorize", "rebuild").
inline const char* toString(UpdateLevel l)
{
    switch (l) {
    case UpdateLevel::Reuse: return "reuse";
    case UpdateLevel::Refactorize: return "refactorize";
    case UpdateLevel::Rebuild: return "rebuild";
    }
    return "?";
}

/**
 * @brief Makes a level decided on every rank identical on all ranks (collective, maximum).
 *
 * Decisions based on floating-point numbers (e.g. GMRES residuals) can differ in the last bit
 * between ranks; a rank refactorizing while another one rebuilds would deadlock in the
 * collectives of the coarse solver. The most expensive request wins.
 */
inline UpdateLevel agree(UpdateLevel l, MPI_Comm comm)
{
    int v = static_cast<int>(l), out = 0;
    MPI_Allreduce(&v, &out, 1, MPI_INT, MPI_MAX, comm);
    return static_cast<UpdateLevel>(out);
}

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

    /**
     * @brief Brings the preconditioner up to date with the current matrix values (collective).
     *
     * Call it after part::Subdomain::updateValues(). The requested level is first made equal on
     * all ranks (agree()). A Refactorize request on a preconditioner that is already up to date
     * does nothing and returns Reuse; Rebuild is always performed.
     * @param requested Level asked by the caller (e.g. by a ReusePolicy).
     * @return The level actually performed.
     */
    virtual UpdateLevel update(UpdateLevel /*requested*/) { return UpdateLevel::Reuse; }

    /// @brief True if the preconditioner was built for the current matrix values.
    virtual bool upToDate() const { return true; }

    /// @brief Local numerical refactorizations of this rank that fell back to a full factorization.
    virtual long localFallbacks() const { return 0; }
    /// @brief Coarse numerical refactorizations that fell back to a full factorization.
    virtual long coarseFallbacks() const { return 0; }
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
