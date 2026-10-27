/**
 * @file krylov.hpp
 * @brief Common interface of the Krylov solvers (GMRES, BiCGStab): stopping norms, result, factory.
 */
#ifndef SCHWARZ2LVL_SOLVER_KRYLOV_HPP
#define SCHWARZ2LVL_SOLVER_KRYLOV_HPP

#include "config.hpp"
#include "distributed_matrix.hpp"
#include "preconditioner.hpp"
#include "types.hpp"

#include <mpi.h>

#include <memory>
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

/// @brief Outcome of a Krylov solve.
struct KrylovResult {
    bool converged = false;          ///< rtol reached.
    bool breakdown = false;          ///< BiCGStab: (r^, A y) = 0 (the method cannot continue).
    int iterations = 0;              ///< Applications of M^{-1}, each with one product with A (GMRES: Arnoldi
                                     ///< steps; BiCGStab: two per BiCGStab step, one if it stops at a half step).
    int restarts = 0;                ///< BiCGStab: restarts (shadow residual orthogonal to r, or true residual check).
    double rel_residual = 0.0;       ///< Last residual of the iteration (GMRES estimate, BiCGStab recursive) / ||b||_2.
    double true_rel_residual = 0.0;  ///< ||b - A x||_2 / ||b||_2 recomputed at the end.
    double initial_residual_of = 0.0;///< OpenFOAM normalised residual of x0 (StopNorm::OpenFOAM only).
    double final_residual_of = 0.0;  ///< OpenFOAM normalised residual of x, recomputed (StopNorm::OpenFOAM only).
    std::vector<double> history;     ///< ||r||_2 / ||b||_2 after every application of M^{-1} (entry 0: initial).
};

/**
 * @class KrylovSolver
 * @brief Solves A x = b with a preconditioner M^{-1} (collective).
 */
class KrylovSolver {
public:
    virtual ~KrylovSolver() = default;

    /**
     * @brief Solves A x = b (collective).
     * @param A Distributed operator.
     * @param M Preconditioner.
     * @param b Owned entries of the right-hand side.
     * @param x Owned entries of the initial guess on input, of the solution on output.
     */
    virtual KrylovResult solve(const linalg::DistributedMatrix& A, const prec::Preconditioner& M, const core::Vec& b,
                               core::Vec& x) const = 0;

    /// @brief Short name, e.g. "GMRES(30)" or "BiCGStab".
    virtual std::string name() const = 0;
};

/// @brief Krylov solver of a configuration: --krylov gmres (with --restart) or bicgstab; --rtol, --maxit, --stop-norm.
std::unique_ptr<KrylovSolver> makeKrylovSolver(const core::SolverConfig& cfg);

namespace detail {

/// @brief sum_i |v_i| over all ranks.
double norm1(const core::Vec& v, MPI_Comm comm);

/// @brief OpenFOAM normalisation factor (lduMatrix::solver::normFactor) of the initial guess x0:
///        F = sum|A x0 - xm A 1| + sum|b - xm A 1| + 1e-20, xm = mean of x0 over all rows.
/// @param Ax0 A x0 (already computed by the caller).
double openfoamNormFactor(const linalg::DistributedMatrix& A, const core::Vec& b, const core::Vec& x0, const core::Vec& Ax0,
                          MPI_Comm comm);

} // namespace detail

} // namespace schwarz2lvl::solv

#endif // SCHWARZ2LVL_SOLVER_KRYLOV_HPP
