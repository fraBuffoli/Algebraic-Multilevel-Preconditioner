/**
 * @file local_eigensolver.hpp
 * @brief Local generalized eigenproblem (3.1) and local coarse basis Z_i of (3.2).
 */
#ifndef SCHWARZ2LVL_PRECONDITIONER_LOCAL_EIGENSOLVER_HPP
#define SCHWARZ2LVL_PRECONDITIONER_LOCAL_EIGENSOLVER_HPP

#include "types.hpp"

#include <complex>
#include <string>
#include <vector>

namespace schwarz2lvl::prec {

/// @brief Parameters of the local eigensolver (see SolverConfig for the meaning).
struct LocalEigensolverOptions {
    double tau = 0.6;                   ///< Select |lambda| >= 1/tau.
    int nev = 300;                      ///< Number of largest-magnitude eigenpairs computed (incl. the kernel part).
    double tol = 1e-8;                  ///< Relative tolerance of the Arnoldi method.
    int maxit = 1000;                   ///< Maximum number of restarts.
    int ncv = 0;                        ///< Krylov subspace size (0 = automatic).
    double shift_rel = 1e-13;           ///< sigma / ||Atilda_ii|| used to factorize Atilda_ii + sigma I.
    double kernel_tol = 1e-8;           ///< Relative threshold of the numerical kernels.
    int kernel_probe = 8;               ///< Initial block size of the kernel detection.
    int dense_threshold = 400;          ///< n_i <= threshold: dense (reference) path.
    std::string local_solver = "auto";  ///< Backend used to factorize Atilda_ii + sigma I.
    bool symmetric = false;             ///< A (hence Atilda_ii) is symmetric: the left kernel equals the right kernel.
};

/// @brief Output of LocalEigensolver::compute().
struct LocalEigensolverResult {
    core::Mat Z;                                    ///< Basis of (3.2): [ (L∩K)^⊥_K | selected eigenvectors ].
    int kernel_dim = 0;                             ///< dim K_i = dim ker(Atilda_ii).
    int kernel_part = 0;                            ///< dim (L_i ∩ K_i)^⊥_{K_i}: first columns of Z.
    int n_computed = 0;                             ///< Converged eigenpairs of the regular part.
    int n_selected = 0;                             ///< Columns of Z coming from eigenvectors (|lambda| >= 1/tau).
    std::vector<std::complex<double>> eigenvalues;  ///< Computed finite eigenvalues (decreasing modulus).
    bool dense = false;                             ///< The dense path was used.
    bool converged = true;                          ///< All requested eigenpairs converged.
    double shift = 0.0;                             ///< Shift sigma actually used (sparse path).
    long matvecs = 0;                               ///< Applications of the projected operator (sparse path).
    std::string backend;                            ///< Factorization backend (sparse path).
};

/**
 * @class LocalEigenSolver
 * @brief Computes Z_i of (3.2) for one subdomain.
 *
 * Problem (3.1): \f$ \Pi_i B_i \Pi_i u = \lambda \tilde A_{ii} u \f$ with
 * \f$ B_i = D_i A_{ii} D_i \f$, \f$ L_i = \ker B_i \f$, \f$ K_i = \ker\tilde A_{ii} \f$,
 * and Z_i spanning \f$ (L_i\cap K_i)^{\perp_{K_i}} \oplus \mathrm{span}\{u : |\lambda|\ge 1/\tau\} \f$.
 *
 * Atilda_ii is **singular** for every subdomain whose rows have zero sum after
 * lumping (floating subdomains of a conservative discretization: Atilda_ii 1 = 0),
 * so it can never be factorized as it is. The algorithm separates the
 * singular and the regular part of the pencil explicitly:
 *
 *  1. Factorize \f$ \tilde A_{ii} + \sigma I \f$, sigma = shift_rel * ||Atilda_ii||
 *     (a regularization at the level of the rounding errors).
 *  2. **Kernel K_i** (right): block inverse iteration with (A~+sigma I)^{-1}
 *     (3 steps, block size kernel_probe, doubled while every probe vector is
 *     in the kernel), then an SVD of A~ Y; singular values below
 *     kernel_tol * ||A~|| define K_i = range(Q). If A is nonsymmetric the
 *     **left** kernel P (ker A~^T) is computed in the same way with
 *     transposed solves, because range(A~) = P^⊥ differs from Q^⊥.
 *  3. **Infinite eigenvalues**: (L∩K)^⊥_K = Q * (right singular vectors of
 *     B Q with nonzero singular values). These are the eigenvectors whose
 *     eigenvalues are "close to eps^{-1} in inexact arithmetic" in the paper;
 *     here they are computed exactly and always included.
 *  4. **Finite eigenvalues**: restricted to K^⊥, (3.1) is equivalent to
 *     \f$ \tilde A^+ \Pi_R B \Pi_K u = \lambda u \f$ with
 *     \f$\Pi_K = I - QQ^T\f$ and \f$\Pi_R = I - PP^T\f$ (orthogonal projection on
 *     range(A~)). Since \f$ \Pi_K (\tilde A+\sigma I)^{-1}\Pi_R = \tilde A^+ + O(\sigma) \f$
 *     without ever producing O(1/sigma) components, the operator
 *     \f$ T = \Pi_K (\tilde A+\sigma I)^{-1} \Pi_R B \Pi_K \f$ is applied
 *     matrix-free inside the implicitly restarted Arnoldi method of Spectra
 *     (LargestMagn). One sparse triangular solve + one SpMV per iteration.
 *  5. **Selection**: among the nev - dim((L∩K)^⊥_K) largest-modulus
 *     eigenvalues, those with |lambda| >= 1/tau (Section 4: "compute the
 *     nev largest, keep those verifying |lambda| >= 1/tau"). A complex pair
 *     contributes the real and imaginary parts of its eigenvector (same real
 *     invariant subspace, real arithmetic).
 *
 * For n_i <= dense_threshold (or when the Krylov space would not be smaller
 * than n_i) the same quantities are computed densely: SVD of A~ (kernels and
 * pseudo-inverse), T = A~^+ B Pi_K, full eigendecomposition. This path is
 * also the reference used by the tests.
 */
class LocalEigensolver {
public:
    /// @brief Stores the options.
    explicit LocalEigensolver(LocalEigensolverOptions opt) : opt_(std::move(opt)) {}

    /**
     * @brief Computes Z_i (purely local, no communication).
     * @param Aii    Local matrix A_ii.
     * @param Atilde Local block splitting matrix Atilda_ii.
     * @param D      Diagonal of the partition of unity D_i.
     * @param force_dense Use the dense path regardless of the size.
     * @return Basis and statistics.
     * @throws std::runtime_error if Atilda_ii + sigma I cannot be factorized.
     */
    LocalEigensolverResult compute(const core::SpMat& Aii, const core::SpMat& Atilde, const core::Vec& D, bool force_dense = false) const;

    /// @brief Options.
    const LocalEigensolverOptions& options() const { return opt_; }

private:
    LocalEigensolverOptions opt_; ///< Options.

    /// @brief Dense reference path.
    LocalEigensolverResult computeDense(const core::SpMat& Atilde, const core::SpMat& B) const;
    /// @brief Sparse (Spectra) path.
    LocalEigensolverResult computeSparse(const core::SpMat& Atilde, const core::SpMat& B) const;
};

} // namespace schwarz2lvl::prec

#endif // SCHWARZ2LVL_PRECONDITIONER_LOCAL_EIGENSOLVER_HPP
