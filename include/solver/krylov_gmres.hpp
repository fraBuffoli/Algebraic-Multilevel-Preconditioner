#ifndef KRYLOV_GMRES_HPP
#define KRYLOV_GMRES_HPP

#include "krylov_solver.hpp"

namespace schwarz2lvl {

/**
 * @class KrylovGmres
 * @brief Implements the Generalized Minimal Residual (GMRES) iterative solver with restart.
 * 
 * This class inherits from KrylovSolver and implements the right-preconditioned 
 * GMRES method using Givens rotations for on-the-fly residual norm tracking.
 */
class KrylovGmres : public KrylovSolver {
public:
    /**
     * @brief Constructs the GMRES solver.
     * 
     * @param max_iter Maximum total number of linear iterations allowed.
     * @param tolerance Relative residual tolerance for the stopping criterion.
     * @param restart_dim Dimension of the Krylov subspace before performing a restart (m).
     */
    KrylovGmres(int max_iter, double tolerance, int restart_dim) : 
        KrylovSolver(max_iter, tolerance), restart_dim_(restart_dim) {};

    /**
     * @brief Destructor.
     */
    virtual ~KrylovGmres() = default;

    /**
     * @brief Executes the right-preconditioned GMRES solution loop: A * M^-1 * (M * x) = b.
     * 
     * @param A The global sparse matrix wrapper.
     * @param b The right-hand side global vector.
     * @param x The solution global vector (initial guess and final output).
     * @param prec The preconditioner object (abstract interface).
     * @return true if convergence is achieved, false otherwise.
     */
    virtual bool solve(const SparseMatrixWrapper& A,
                       const VectorType& b,
                       VectorType& x,
                       const Preconditioner& prec) const override;

private:
    int restart_dim_; // Subspace dimension before restart (m)
};

} // namespace schwarz2lvl

#endif // KRYLOV_GMRES_HPP
