/**
 * @file gmres.hpp
 * @brief Right-preconditioned restarted GMRES(m) on distributed vectors.
 */
#ifndef SCHWARZ2LVL_SOLVER_GMRES_HPP
#define SCHWARZ2LVL_SOLVER_GMRES_HPP

#include "distributed_matrix.hpp"
#include "preconditioner.hpp"
#include "types.hpp"

#include <string>
#include <vector>

namespace schwarz2lvl::solv {

/**
 * @brief Norm of the stopping test.
 *
 *  - L2: ||b - A x||_2 <= rtol ||b||_2 (Section 4 of the paper; default).
 *  - OpenFOAM: the normalised residual of the OpenFOAM linear solvers (lduMatrix::solver):
 *    sum|b - A x| / F, F = sum|A x0 - xm A 1| + sum|b - xm A 1| + 1e-20 (xm = mean of x0), and
 *    relTol semantics: stop when it is <= rtol times its initial value, i.e.
 *    sum|b - A x| <= rtol sum|b - A x0| (from x0 = 0: sum|b - A x| / sum|b| <= rtol). The same test
 *    as OpenFOAM with `tolerance 0; relTol rtol;`, for comparisons and for an OpenFOAM port.
 */
enum class StopNorm { L2, OpenFOAM };

/// @brief "l2" | "openfoam" -> StopNorm (throws std::invalid_argument otherwise).
StopNorm parseStopNorm(const std::string& s);
/// @brief StopNorm -> "l2" | "openfoam".
const char* toString(StopNorm s);

/// @brief Outcome of a GMRES solve.
struct GmresResult {
    bool converged = false;          ///< rtol reached.
    int iterations = 0;              ///< Total Arnoldi steps.
    double rel_residual = 0.0;       ///< Last GMRES residual estimate / ||b||.
    double true_rel_residual = 0.0;  ///< ||b - A x||_2 / ||b||_2 recomputed at the end.
    double initial_residual_of = 0.0;///< OpenFOAM normalised residual of x0 (StopNorm::OpenFOAM only).
    double final_residual_of = 0.0;  ///< OpenFOAM normalised residual of x, recomputed (StopNorm::OpenFOAM only).
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
 * Convergence is accepted only if the true residual ||b - A x||, recomputed at the end of
 * the cycle, is within 10% of the tolerance; otherwise GMRES restarts from it (relevant
 * when the preconditioner is applied in single precision).
 *
 * With StopNorm::OpenFOAM the test is on sum|r|: the GMRES residual of every iteration,
 * r = g_{j+1} V_{j+1} Q^T e_{j+1} (Q = product of the Givens rotations), is formed with one pass
 * over the basis and one reduction (timer solve.stopping_test).
 */
class Gmres {
public:
    /**
     * @param restart Restart length m.
     * @param rtol    Relative tolerance.
     * @param maxit   Maximum number of iterations.
     */
    Gmres(int restart, double rtol, int maxit, StopNorm stop = StopNorm::L2)
        : m_(restart), rtol_(rtol), maxit_(maxit), stop_(stop)
    {
    }

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
    StopNorm stop_;///< Norm of the stopping test.
};

} // namespace schwarz2lvl::solv

#endif // SCHWARZ2LVL_SOLVER_GMRES_HPP
