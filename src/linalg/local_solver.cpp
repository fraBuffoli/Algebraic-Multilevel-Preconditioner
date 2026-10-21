/**
 * @file local_solver.cpp
 * @brief Implementation of the local direct solver backends.
 */
#include "local_solver.hpp"

#include <Eigen/SparseLU>

#ifdef USE_UMFPACK
#include <umfpack.h>
#endif
#ifdef USE_PARDISO
#include <Eigen/PardisoSupport>
#endif

#include <stdexcept>

namespace schwarz2lvl::linalg {

void LocalSolver::solve(const core::Mat& B, core::Mat& X) const
{
    X.resize(B.rows(), B.cols());
    core::Vec x;
    for (Eigen::Index j = 0; j < B.cols(); ++j) {
        solve(core::Vec(B.col(j)), x);
        X.col(j) = x;
    }
}

void LocalSolver::solveTranspose(const core::Vec&, core::Vec&) const
{
    throw std::logic_error("solveTranspose not supported by backend " + name());
}

void LocalSolver::solveTranspose(const core::Mat& B, core::Mat& X) const
{
    X.resize(B.rows(), B.cols());
    core::Vec x;
    for (Eigen::Index j = 0; j < B.cols(); ++j) {
        solveTranspose(core::Vec(B.col(j)), x);
        X.col(j) = x;
    }
}

// --------------------------------------------------------------------- SparseLU
/**
 * @brief Eigen::SparseLU backend (supernodal LU with partial pivoting, COLAMD ordering).
 */
class SparseLuSolver final : public LocalSolver {
public:
    bool factorize(const core::SpMat& A) override
    {
        lu_.analyzePattern(A);
        lu_.factorize(A);
        if (lu_.info() != Eigen::Success) {
            error_ = "Eigen::SparseLU: " + lu_.lastErrorMessage();
            return false;
        }
        return true;
    }
    bool refactorize(const core::SpMat& A) override
    {
        fell_back_ = false;
        lu_.factorize(A); // the column ordering / elimination tree of analyzePattern() are kept
        if (lu_.info() == Eigen::Success) return true;
        fell_back_ = true;
        return factorize(A);
    }
    void solve(const core::Vec& b, core::Vec& x) const override { x = lu_.solve(b); }
    void solve(const core::Mat& B, core::Mat& X) const override { X = lu_.solve(B); }
    bool hasTransposeSolve() const override { return true; }
    void solveTranspose(const core::Vec& b, core::Vec& x) const override
    {
        // SparseLU::transpose() is a non-const view in Eigen 3.4.
        x = const_cast<Eigen::SparseLU<core::SpMat, Eigen::COLAMDOrdering<core::LocalIndex>>&>(lu_).transpose().solve(b);
    }
    std::string name() const override { return "sparselu"; }

private:
    Eigen::SparseLU<core::SpMat, Eigen::COLAMDOrdering<core::LocalIndex>> lu_; ///< Factorization.
};

#ifdef USE_UMFPACK
// ---------------------------------------------------------------------- UMFPACK
/**
 * @brief SuiteSparse UMFPACK backend (unsymmetric multifrontal LU), called directly
 *        on the compressed column arrays of the Eigen matrix (no conversion).
 */
class UmfpackSolver final : public LocalSolver {
public:
    UmfpackSolver()
    {
        umfpack_di_defaults(control_);
        // No iterative refinement: inside a preconditioner / eigensolver the LU backward
        // error is sufficient, and refinement (1 SpMV + 1 solve per step, default 2 steps)
        // made each solve ~2.8x slower in the convdiff3d tests.
        control_[UMFPACK_IRSTEP] = 0;
        // Nested-dissection (METIS) fill-reducing ordering instead of the default AMD/COLAMD:
        // for 3D problems it gives much less fill (convdiff3d 36^3: nnz(L+U) -31%,
        // numeric factorization 2.1x faster, solves ~25% faster).
        control_[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
    }
    ~UmfpackSolver() override { release(); }
    UmfpackSolver(const UmfpackSolver&) = delete;
    UmfpackSolver& operator=(const UmfpackSolver&) = delete;

    bool factorize(const core::SpMat& A) override
    {
        release();
        A_ = A; // UMFPACK needs the matrix during the solves (iterative refinement)
        A_.makeCompressed();
        const int n = static_cast<int>(A_.rows());
        int status = umfpack_di_symbolic(n, n, A_.outerIndexPtr(), A_.innerIndexPtr(), A_.valuePtr(), &symbolic_,
                                         control_, info_);
        if (status != UMFPACK_OK) {
            error_ = "UMFPACK symbolic factorization failed (status " + std::to_string(status) + ")";
            return false;
        }
        status = umfpack_di_numeric(A_.outerIndexPtr(), A_.innerIndexPtr(), A_.valuePtr(), symbolic_, &numeric_,
                                    control_, info_);
        if (status != UMFPACK_OK) {
            error_ = "UMFPACK numeric factorization failed (status " + std::to_string(status) +
                     (status == UMFPACK_WARNING_singular_matrix ? ": singular matrix)" : ")");
            return false;
        }
        return true;
    }
    bool refactorize(const core::SpMat& A) override
    {
        fell_back_ = false;
        if (symbolic_ == nullptr) return factorize(A);
        if (numeric_) umfpack_di_free_numeric(&numeric_);
        numeric_ = nullptr;
        A_ = A; // same pattern: only the values are new
        A_.makeCompressed();
        const int status = umfpack_di_numeric(A_.outerIndexPtr(), A_.innerIndexPtr(), A_.valuePtr(), symbolic_,
                                              &numeric_, control_, info_); // symbolic object reused
        if (status == UMFPACK_OK) return true;
        fell_back_ = true;
        return factorize(A);
    }
    void solve(const core::Vec& b, core::Vec& x) const override { run(UMFPACK_A, b, x); }
    bool hasTransposeSolve() const override { return true; }
    void solveTranspose(const core::Vec& b, core::Vec& x) const override { run(UMFPACK_At, b, x); }
    std::string name() const override { return "umfpack"; }

private:
    core::SpMat A_;                          ///< Copy of the factorized matrix (required by the UMFPACK API).
    void* symbolic_ = nullptr;         ///< UMFPACK symbolic object.
    void* numeric_ = nullptr;          ///< UMFPACK numeric object.
    double control_[UMFPACK_CONTROL];  ///< UMFPACK control parameters.
    mutable double info_[UMFPACK_INFO];///< UMFPACK statistics.

    void run(int sys, const core::Vec& b, core::Vec& x) const
    {
        x.resize(b.size());
        const int status = umfpack_di_solve(sys, A_.outerIndexPtr(), A_.innerIndexPtr(), A_.valuePtr(), x.data(),
                                            b.data(), numeric_, const_cast<double*>(control_), info_);
        if (status != UMFPACK_OK && status != UMFPACK_WARNING_singular_matrix)
            throw std::runtime_error("UMFPACK solve failed (status " + std::to_string(status) + ")");
    }
    void release()
    {
        if (symbolic_) umfpack_di_free_symbolic(&symbolic_);
        if (numeric_) umfpack_di_free_numeric(&numeric_);
        symbolic_ = numeric_ = nullptr;
    }
};
#endif

#ifdef USE_PARDISO
// ---------------------------------------------------------------------- PARDISO
/**
 * @brief Intel MKL PARDISO backend through Eigen::PardisoLU (one thread per MPI rank:
 *        set MKL_NUM_THREADS=1). Transposed solves use iparm(12) = 2.
 *
 * No iterative refinement (iparm(8) = 0; Eigen sets 2): inside a preconditioner and an
 * eigensolver the backward error of the LU is sufficient, and the two refinement steps
 * (1 SpMV + 1 solve each, always performed) made every solve 3-3.7x slower. PARDISO still
 * refines automatically (2 steps) the solves of a factorization with perturbed pivots.
 */
class PardisoSolver final : public LocalSolver {
public:
    PardisoSolver() { lu_.pardisoParameterArray()[7] = 0; } // kept by compute(): set once

    bool factorize(const core::SpMat& A) override
    {
        lu_.compute(A); // phase 12: analysis (ordering, scaling + matching from the values) + factorization
        if (lu_.info() != Eigen::Success) {
            error_ = "PARDISO factorization failed";
            return false;
        }
        perturbed_full_ = perturbedPivots();
        return true;
    }
    /// Phase 22 only. For nonsymmetric matrices the scaling and the weighted matching of the
    /// analysis come from the values of the last full factorization (MKL: they are computed in
    /// phase 11); if they no longer fit, PARDISO perturbs more pivots: fall back to phase 12.
    bool refactorize(const core::SpMat& A) override
    {
        fell_back_ = false;
        lu_.factorize(A);
        if (lu_.info() == Eigen::Success && perturbedPivots() <= perturbed_full_) return true;
        fell_back_ = true;
        return factorize(A);
    }
    void solve(const core::Vec& b, core::Vec& x) const override { x = lu_.solve(b); }
    void solve(const core::Mat& B, core::Mat& X) const override { X = lu_.solve(B); }
    bool hasTransposeSolve() const override { return true; }
    void solveTranspose(const core::Vec& b, core::Vec& x) const override
    {
        lu_.pardisoParameterArray()[11] = 2; // solve with the transposed matrix
        x = lu_.solve(b);
        lu_.pardisoParameterArray()[11] = 0;
    }
    void solveTranspose(const core::Mat& B, core::Mat& X) const override // one call, nrhs = B.cols()
    {
        lu_.pardisoParameterArray()[11] = 2;
        X = lu_.solve(B);
        lu_.pardisoParameterArray()[11] = 0;
    }
    long perturbedPivots() const override { return static_cast<long>(lu_.pardisoParameterArray()[13]); }
    std::string name() const override { return "pardiso"; }

private:
    mutable Eigen::PardisoLU<core::SpMat> lu_; ///< Factorization (mutable: iparm is changed for A^T solves).
    long perturbed_full_ = 0;                  ///< Perturbed pivots of the last full factorization.
};
#endif

std::vector<std::string> availableLocalSolvers()
{
    std::vector<std::string> v;
#ifdef USE_PARDISO
    v.push_back("pardiso");
#endif
#ifdef USE_UMFPACK
    v.push_back("umfpack");
#endif
    v.push_back("sparselu");
    return v;
}

std::unique_ptr<LocalSolver> makeLocalSolver(const std::string& name)
{
    const std::string n = (name == "auto") ? availableLocalSolvers().front() : name;
    if (n == "sparselu") return std::make_unique<SparseLuSolver>();
#ifdef USE_UMFPACK
    if (n == "umfpack") return std::make_unique<UmfpackSolver>();
#endif
#ifdef USE_PARDISO
    if (n == "pardiso") return std::make_unique<PardisoSolver>();
#endif
    throw std::invalid_argument("local solver '" + n + "' is unknown or not compiled in");
}

} // namespace schwarz2lvl