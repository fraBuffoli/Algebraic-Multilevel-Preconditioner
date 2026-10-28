/**
 * @file gmres.hpp
 * @brief Right-preconditioned restarted GMRES(m) on distributed vectors.
 */
#ifndef SCHWARZ2LVL_SOLVER_GMRES_HPP
#define SCHWARZ2LVL_SOLVER_GMRES_HPP

#include "krylov.hpp"

#include <string>

namespace schwarz2lvl::solv {

/// @brief Outcome of a GMRES solve (same fields as any Krylov solve).
using GmresResult = KrylovResult;

/**
 * @class Gmres
 * @brief Restarted GMRES with right preconditioning, as in the paper's experiments.
 *
 * Solves A M^{-1} u = b, x = M^{-1} u. With right preconditioning the GMRES
 * residual is the *unpreconditioned* residual b - A x, so the stopping test
 * ||b - A x|| <= rtol ||b|| is the one of Section 4.
 *
 * Orthogonalization: classical Gram-Schmidt applied twice (CGS2): as stable
 * as modified Gram-Schmidt in practice, but the j inner products of a pass
 * are done with a single MPI_Allreduce (2 reductions per iteration instead
 * of j+1), which matters at large process counts. Givens rotations update the
 * least-squares residual. M^{-1} is applied once per iteration plus once per
 * restart cycle (x += M^{-1} V y), so only the Krylov basis V is stored.
 * Convergence is accepted only if the true residual ||b - A x||, recomputed at the end of
 * the cycle, is within 10% of the tolerance; otherwise GMRES restarts from it (relevant
 * when the preconditioner is applied in single precision).
 *
 * With StopNorm::OpenFOAM the test is on sum|r|: the GMRES residual of every iteration,
 * r = g_{j+1} V_{j+1} Q^T e_{j+1} (Q = product of the Givens rotations), is formed with one pass
 * over the basis and one reduction (timer solve.stopping_test).
 */
class Gmres final : public KrylovSolver {
public:
    /**
     * @param restart Restart length m.
     * @param rtol    Relative tolerance.
     * @param maxit   Maximum number of iterations.
     * @param stop    Norm of the stopping test.
     * @param atol    StopNorm::OpenFOAM: absolute tolerance on the normalised residual (0 = off).
     */
    Gmres(int restart, double rtol, int maxit, StopNorm stop = StopNorm::L2, double atol = 0.0)
        : m_(restart), rtol_(rtol), atol_(atol), maxit_(maxit), stop_(stop)
    {
    }

    /**
     * @brief Solves A x = b (collective).
     * @param A Distributed operator.
     * @param M Preconditioner (applied on the right).
     * @param b Owned entries of the right-hand side.
     * @param x Owned entries of the initial guess on input, of the solution on output.
     */
    KrylovResult solve(const linalg::DistributedMatrix& A, const prec::Preconditioner& M, const core::Vec& b,
                       core::Vec& x) const override;

    std::string name() const override { return "GMRES(" + std::to_string(m_) + ")"; }

private:
    int m_;        ///< Restart length.
    double rtol_;  ///< Relative tolerance.
    double atol_;  ///< Absolute tolerance on the OpenFOAM normalised residual.
    int maxit_;    ///< Maximum iterations.
    StopNorm stop_;///< Norm of the stopping test.
};

} // namespace schwarz2lvl::solv

#endif // SCHWARZ2LVL_SOLVER_GMRES_HPP
