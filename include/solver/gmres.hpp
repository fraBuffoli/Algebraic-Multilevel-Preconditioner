/**
 * @file gmres.hpp
 * @brief Right-preconditioned restarted GMRES(m) on distributed vectors.
 */
#ifndef SCHWARZ2LVL_SOLVER_GMRES_HPP
#define SCHWARZ2LVL_SOLVER_GMRES_HPP

#include "distributed_matrix.hpp"
#include "preconditioner.hpp"
#include "types.hpp"

#include <vector>

namespace schwarz2lvl::solv {

/// @brief Outcome of a GMRES solve.
struct GmresResult {
    bool converged = false;          ///< rtol reached.
    int iterations = 0;              ///< Total Arnoldi steps.
    double rel_residual = 0.0;       ///< Last GMRES residual estimate / ||b||.
    double true_rel_residual = 0.0;  ///< ||b - A x|| / ||b|| recomputed at the end.
    std::vector<double> history;     ///< Residual estimate / ||b|| at every iteration (entry 0: initial).
};

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
 */
class Gmres {
public:
    /**
     * @param restart Restart length m.
     * @param rtol    Relative tolerance.
     * @param maxit   Maximum number of iterations.
     */
    Gmres(int restart, double rtol, int maxit) : m_(restart), rtol_(rtol), maxit_(maxit) {}

    /**
     * @brief Solves A x = b (collective).
     * @param A Distributed operator.
     * @param M Preconditioner (applied on the right).
     * @param b Owned entries of the right-hand side.
     * @param x Owned entries of the initial guess on input, of the solution on output.
     */
    GmresResult solve(const linalg::DistributedMatrix& A, const prec::Preconditioner& M, const core::Vec& b, core::Vec& x) const;

private:
    int m_;        ///< Restart length.
    double rtol_;  ///< Relative tolerance.
    int maxit_;    ///< Maximum iterations.
};

} // namespace schwarz2lvl::solv

#endif // SCHWARZ2LVL_SOLVER_GMRES_HPP
