/**
 * @file time_loop.hpp
 * @brief Sequence of solves with the same sparsity pattern (a CFD-like time loop) and setup reuse.
 */
#ifndef SCHWARZ2LVL_SOLVER_TIME_LOOP_HPP
#define SCHWARZ2LVL_SOLVER_TIME_LOOP_HPP

#include "config.hpp"
#include "preconditioner.hpp"
#include "types.hpp"

#include <mpi.h>

#include <iosfwd>
#include <string>
#include <vector>

namespace schwarz2lvl::solv {

/// @brief Outcome of one solve of the time loop (identical on every rank).
struct SolveRecord {
    int step = 0;                     ///< Time step.
    int corrector = 0;                ///< Corrector inside the step (same matrix; new right-hand side if random).
    prec::UpdateLevel level = prec::UpdateLevel::Reuse; ///< Setup work done before the solve.
    double t_update = 0.0;            ///< Wall time of the setup work [s] (initial setup for the first solve).
    double t_solve = 0.0;             ///< Wall time of the Krylov solve [s].
    int iterations = 0;               ///< Applications of the preconditioner.
    double per_decade = 0.0;          ///< Iterations per decade of residual reduction.
    bool converged = false;           ///< rtol reached.
    double true_rel_residual = 0.0;   ///< ||b - A x|| / ||b||.
    core::GlobalIndex n0 = 0;         ///< Coarse dimension after the update (0 for one-level).
    double wasted = 0.0;              ///< ReusePolicy::wasted() after the solve [s].
    double initial_res_of = 0.0;      ///< OpenFOAM normalised residual of the initial guess (--stop-norm openfoam).
    double final_res_of = 0.0;        ///< OpenFOAM normalised residual of the solution (--stop-norm openfoam).
    bool final_corrector = false;     ///< Last corrector of the step (solved with --final-rtol / --final-atol).
};

/**
 * @brief Runs `cfg.steps` steps of `cfg.correctors` solves each (collective).
 *
 * The emulated situation is the pressure equation of a PISO/PIMPLE loop:
 *  - step s: the matrix values change (same pattern): DomainDecomposer::updateValues();
 *  - corrector c: the matrix is the same; with --rhs random the right-hand side is new (generated
 *    in the original ordering so that it does not depend on the number of processes), with
 *    --rhs generated it is the one of the step;
 *  - before each solve: Subdomain::updateValues() (cheap when nothing changed), then the
 *    ReusePolicy chooses the level and Preconditioner::update() performs it;
 *  - the Krylov solver (right preconditioning) from a zero initial guess (comparable iteration
 *    counts), or with --warm-start from the solution of the previous solve, as OpenFOAM does; the
 *    last corrector of a step uses --final-rtol / --final-atol when given (p_rghFinal).
 * The first solve builds the preconditioner (its t_update is the initial setup).
 * Every solve is printed on @p out (rank 0) and, if `cfg.steps_csv` is set, appended to it.
 * @return One record per solve, in order.
 */
std::vector<SolveRecord> runTimeLoop(const core::SolverConfig& cfg, MPI_Comm comm, std::ostream& out);

/// @brief Prints totals per level, setup/solve times and iterations of a time loop (rank 0 prints).
void printTimeLoopSummary(const std::vector<SolveRecord>& recs, const core::SolverConfig& cfg, MPI_Comm comm,
                          std::ostream& out);

} // namespace schwarz2lvl::solv

#endif // SCHWARZ2LVL_SOLVER_TIME_LOOP_HPP
