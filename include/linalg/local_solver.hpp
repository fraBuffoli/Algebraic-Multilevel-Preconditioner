/**
 * @file local_solver.hpp
 * @brief Sequential sparse direct solvers for the subdomain matrices.
 *
 * The paper computes *exact* factorizations of the local matrices A_ii
 * (Section 4). Three interchangeable backends are provided behind one
 * interface, selected at run time with `--local-solver`:
 *  - `sparselu` : Eigen built-in solvers (always available): SparseLU (supernodal LU, COLAMD)
 *                 for LU, SimplicialLLT / SimplicialLDLT (AMD) for the symmetric factorizations;
 *  - `umfpack`  : SuiteSparse UMFPACK called directly (compile with USE_UMFPACK=1), LU only;
 *  - `pardiso`  : Intel MKL PARDISO through Eigen (compile with USE_PARDISO=1, the natural choice
 *                 on CINECA G100): PardisoLU (mtype 11), PardisoLLT (mtype 2, symmetric positive
 *                 definite), PardisoLDLT (mtype -2, symmetric indefinite, Bunch-Kaufman pivoting).
 *
 * **Factorization type** (`--local-factorization`). For a symmetric matrix a symmetric
 * factorization stores one triangular factor instead of two (A = L L^T or L D L^T instead of
 * A = L U): half the memory and about half the work of the (re)factorization. A solve still
 * reads the factor twice (forward with L, backward with L^T), as many bytes as L and U: it
 * gains only from the smaller footprint (~20% measured on a 32^3 subdomain with PARDISO).
 * LDL^T with Bunch-Kaufman pivoting is not faster than LU in PARDISO: it is only a fallback.
 * Cholesky needs a positive definite matrix (A_ii of an SPD A is SPD; A~_ii + sigma I is PSD +
 * shift; A_00 = V^T A V is SPD); LDL^T accepts symmetric indefinite matrices. The request
 * `auto` builds a chain Cholesky -> LDL^T -> LU: each candidate is tried until one factorizes
 * successfully, and the chain stays on the one that worked. A symmetric factorization only
 * reads one triangle, so it would be silently wrong on a nonsymmetric matrix: for a matrix
 * not known to be symmetric every request becomes LU.
 *
 * **Precision** (`--local-precision`). `single` stores the factors in float (PARDISO
 * iparm[27] = 1, Eigen solvers on float matrices): this halves the bytes read by every solve
 * (~45% faster solves than Cholesky in double on a 32^3 subdomain). The right-hand
 * side is rounded to float, solved, and converted back: the result has a relative error of
 * about 1e-7 times the condition number of the matrix. Meant for the one-level part of the
 * preconditioner (GMRES in double precision corrects it), not for the eigensolver.
 *
 * **Approximate inverse** (`--local-approx amg`, P7): instead of factorizing A_ii exactly, the
 * one-level part can apply one AMG V-cycle (see inexact_solver.hpp). The same LocalSolver
 * interface is used: factorize() = setup, refactorize() = setup with new values, solve() = one
 * application.
 */
#ifndef SCHWARZ2LVL_LINALG_LOCAL_SOLVER_HPP
#define SCHWARZ2LVL_LINALG_LOCAL_SOLVER_HPP

#include "types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace schwarz2lvl::linalg {

/// @brief Type of a sparse factorization.
enum class Factorization { LU, Cholesky, LDLT };

/// @brief Short name of a factorization type ("lu", "cholesky", "ldlt").
const char* toString(Factorization f);

/**
 * @class LocalSolver
 * @brief Abstract sequential sparse direct solver (one factorization, many solves).
 */
class LocalSolver {
public:
    virtual ~LocalSolver() = default;

    /**
     * @brief Computes the factorization of @p A (square, column-major, full storage also when symmetric).
     * @param A Matrix to factorize (the backend may keep a copy).
     * @return false if the factorization failed (e.g. numerically singular matrix, or a
     *         Cholesky factorization of a matrix that is not positive definite).
     */
    virtual bool factorize(const core::SpMat& A) = 0;

    /**
     * @brief Numerical refactorization of a matrix with the same sparsity pattern as the last
     *        one passed to factorize(): the symbolic analysis (fill-reducing ordering,
     *        elimination tree, memory of the factors) is reused, only the numbers are recomputed.
     *
     * The default is a full factorize(); the backends override it. If the numerical phase
     * fails, or it is less accurate than the full factorization (PARDISO: more perturbed
     * pivots, e.g. because scaling and matching were computed from the old values), the
     * backend falls back to factorize() and refactorizationFellBack() returns true.
     * @pre factorize() succeeded once, and @p A has the same pattern (not checked).
     * @return false if the (fallback) factorization failed.
     */
    virtual bool refactorize(const core::SpMat& A) { return factorize(A); }

    /// @brief True if the last refactorize() had to perform a full factorization.
    bool refactorizationFellBack() const { return fell_back_; }

    /**
     * @brief Solves A x = b with the current factorization.
     * @param b Right-hand side.
     * @param x Solution (resized).
     */
    virtual void solve(const core::Vec& b, core::Vec& x) const = 0;

    /**
     * @brief Solves A X = B for several right-hand sides.
     * @param B Right-hand sides (one per column).
     * @param X Solutions (resized).
     */
    virtual void solve(const core::Mat& B, core::Mat& X) const;

    /// @brief True if solveTranspose() is available.
    virtual bool hasTransposeSolve() const { return false; }

    /**
     * @brief Solves A^T x = b with the current factorization.
     * @throws std::logic_error if the backend does not support it.
     */
    virtual void solveTranspose(const core::Vec& b, core::Vec& x) const;

    /**
     * @brief Solves A^T X = B for several right-hand sides (default: column by column).
     * @throws std::logic_error if the backend does not support transposed solves.
     */
    virtual void solveTranspose(const core::Mat& B, core::Mat& X) const;

    /// @brief Pivots perturbed by the last factorization (-1 = not reported by the backend).
    ///        PARDISO refines every solve of such a factorization (2 steps).
    virtual long perturbedPivots() const { return -1; }

    /// @brief Name: backend, factorization, precision (e.g. "pardiso cholesky", "pardiso lu single").
    virtual std::string name() const = 0;

    /// @brief Factorization type in use.
    virtual Factorization factorization() const { return Factorization::LU; }

    /// @brief Position of the factorization in use in its fallback chain (0 = first choice).
    virtual int chainLevel() const { return 0; }

    /// @brief Description of the last failure.
    const std::string& lastError() const { return error_; }

protected:
    std::string error_;      ///< Last error message.
    bool fell_back_ = false; ///< The last refactorize() performed a full factorization.
};

/**
 * @brief Options of the algebraic multigrid approximation of A_ii^{-1} (`--local-approx amg`, P7).
 *
 * One application = one V-cycle from a zero initial guess with `sweeps` Gauss-Seidel steps before
 * and after the coarse correction: a fixed linear operator (same result for the same right-hand
 * side), as the Schwarz preconditioner inside GMRES requires.
 */
struct AmgOptions {
    std::string coarsening = "sa"; ///< "sa" (smoothed aggregation) or "rs" (classical Ruge-Stuben).
    int sweeps = 1;                ///< Gauss-Seidel steps before and after the coarse correction, per level.
    int coarse_enough = 500;       ///< Coarsest level: at most this many unknowns (skyline LU).
    bool reuse = false;            ///< refactorize(): keep the transfer operators, recompute only R A P.
};

/// @brief What to build: backend, factorization request, symmetry of the matrix, precision.
struct LocalSolverSpec {
    std::string backend = "auto";     ///< "auto" (pardiso > umfpack > sparselu), "sparselu", "umfpack", "pardiso".
    std::string factorization = "lu"; ///< "auto", "lu", "cholesky", "ldlt".
    bool symmetric = false;           ///< The matrix is known to be symmetric (otherwise always LU).
    bool single = false;              ///< Single-precision factors.
    /// Approximation of the inverse instead of an exact factorization (P7): "none" or "amg"
    /// (AMGCL V-cycle, build with USE_AMGCL=1). With "amg", backend and factorization are ignored.
    std::string approx = "none";
    AmgOptions amg;                   ///< Used with approx = "amg".
};

/**
 * @brief Creates a solver from a specification.
 *
 * Symmetric matrix: "auto" and "cholesky" give the chain Cholesky -> LDL^T -> LU, "ldlt" gives
 * LDL^T -> LU, "lu" gives LU. Nonsymmetric matrix: LU whatever the request. Backends without a
 * symmetric factorization (umfpack) or without single precision (umfpack) use what they have
 * (LU, double), as their name() reports. With `spec.approx` = "amg" the solver is an
 * approximate inverse (see inexact_solver.hpp): solve() is then *not* exact.
 * @throws std::invalid_argument if the backend or the request is unknown, or the backend is not compiled in.
 */
std::unique_ptr<LocalSolver> makeLocalSolver(const LocalSolverSpec& spec);

/**
 * @brief Creates a solver backend with an LU factorization in double precision.
 * @param name "auto" (pardiso > umfpack > sparselu, depending on the build), "sparselu", "umfpack" or "pardiso".
 * @return The solver.
 * @throws std::invalid_argument if the backend is unknown or not compiled in.
 */
std::unique_ptr<LocalSolver> makeLocalSolver(const std::string& name);

/// @brief Names of the backends compiled into this build.
std::vector<std::string> availableLocalSolvers();

/// @brief Approximations of A_ii^{-1} compiled into this build ("none", and "amg" with USE_AMGCL=1).
std::vector<std::string> availableLocalApprox();

} // namespace schwarz2lvl::linalg

#endif // SCHWARZ2LVL_LINALG_LOCAL_SOLVER_HPP
