#ifndef KRYLOV_SOLVER_HPP
#define KRYLOV_SOLVER_HPP

#include "global_matrix.hpp"
#include "preconditioner.hpp"
#include "timer.hpp"
#include <string>

namespace schwarz2lvl {

/**
 * @class KrylovSolver
 * @brief Abstract base class defining the interface for all iterative Krylov solvers.
 * 
 * This class provides a polymorphic interface allowing different iterative methods 
 * (like GMRES or BiCGStab) to be swapped transparently. It embeds automatic timing.
 */
class KrylovSolver {
public:
    /**
     * @brief Constructor initializing core stopping criteria parameters.
     * 
     * @param max_iter Maximum total number of linear iterations allowed.
     * @param tolerance Relative residual tolerance for the stopping criterion.
     */
    KrylovSolver(int max_iter, double tolerance) 
        : max_iter_(max_iter), tolerance_(tolerance) {}

    /**
     * @brief Virtual destructor ensuring proper memory cleanup of derived solvers.
     */
    virtual ~KrylovSolver() = default;

    /**
     * @brief Pure virtual method to execute the linear system solution: A * x = b.
     * 
     * Must be overridden by specific iterative algorithms (e.g., GMRES, BiCGStab).
     * 
     * @param A The global sparse matrix wrapper.
     * @param b The right-hand side global vector.
     * @param x The solution global vector (acts as initial guess and stores final result).
     * @param prec The preconditioner object inheriting from the abstract Preconditioner interface.
     * @return true if the method converged below the tolerance, false otherwise.
     */
    virtual bool solve(const SparseMatrixWrapper& A,
                       const VectorType& b,
                       VectorType& x,
                       const Preconditioner& prec) const = 0;

    /**
     * @brief Wraps the solve method with automatic RAII timing registry tracking.
     * 
     * Uses ScopedTimer to profile the precise wall-clock time spent in the iterative loop.
     */
    bool solveWithTiming(const SparseMatrixWrapper& A,
                         const VectorType& b,
                         VectorType& x,
                         const Preconditioner& prec) const {
        // Automatically registers elapsed time under "solver.total_solve"
        ScopedTimer timer("solver.total_solve");
        return this->solve(A, b, x, prec);
    }

    /**
     * @brief Iterations performed during the last solve() call.
     */
    int lastIterations() const { return last_iterations_; }

    /**
     * @brief Relative residual reached during the last solve() call.
     */
    double lastResidual() const { return last_residual_; }

    /**
     * @brief Enables/disables per-iteration console output.
     *        Turn it off when running several solves in a batch comparison.
     */
    void setVerbose(bool v) const { verbose_ = v; }

protected:
    int    max_iter_;   // Maximum allowed total iterations
    double tolerance_;  // Relative convergence tolerance

    // Mutable solver state/reporting: solve() is const by interface contract.
    mutable bool   verbose_          = true;
    mutable int    last_iterations_  = 0;
    mutable double last_residual_    = 1.0;
};

} // namespace schwarz2lvl

#endif // KRYLOV_SOLVER_HPP
