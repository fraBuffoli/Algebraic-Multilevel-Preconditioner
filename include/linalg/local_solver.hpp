/**
 * @file local_solver.hpp
 * @brief Sequential sparse direct solvers for the subdomain matrices.
 *
 * The paper computes *exact* LU factorizations of the local matrices A_ii
 * (Section 4). Three interchangeable backends are provided behind one
 * interface, selected at run time with `--local-solver`:
 *  - `sparselu` : Eigen::SparseLU (always available, supernodal LU, COLAMD ordering);
 *  - `umfpack`  : SuiteSparse UMFPACK called directly (compile with USE_UMFPACK=1);
 *  - `pardiso`  : Intel MKL PARDISO through Eigen::PardisoLU (compile with USE_PARDISO=1,
 *                 the natural choice on CINECA G100).
 */
#ifndef SCHWARZ2LVL_LINALG_LOCAL_SOLVER_HPP
#define SCHWARZ2LVL_LINALG_LOCAL_SOLVER_HPP

#include "types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace schwarz2lvl::linalg {

/**
 * @class LocalSolver
 * @brief Abstract sequential sparse direct solver (one factorization, many solves).
 */
class LocalSolver {
public:
    virtual ~LocalSolver() = default;

    /**
     * @brief Computes the LU factorization of @p A (square, column-major).
     * @param A Matrix to factorize (the backend may keep a copy).
     * @return false if the factorization failed (e.g. numerically singular matrix).
     */
    virtual bool factorize(const core::SpMat& A) = 0;

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

    /// @brief Backend name.
    virtual std::string name() const = 0;

    /// @brief Description of the last failure.
    const std::string& lastError() const { return error_; }

protected:
    std::string error_; ///< Last error message.
};

/**
 * @brief Creates a solver backend.
 * @param name "auto" (pardiso > umfpack > sparselu, depending on the build), "sparselu", "umfpack" or "pardiso".
 * @return The solver.
 * @throws std::invalid_argument if the backend is unknown or not compiled in.
 */
std::unique_ptr<LocalSolver> makeLocalSolver(const std::string& name);

/// @brief Names of the backends compiled into this build.
std::vector<std::string> availableLocalSolvers();

} // namespace schwarz2lvl::linalg

#endif // SCHWARZ2LVL_LINALG_LOCAL_SOLVER_HPP
