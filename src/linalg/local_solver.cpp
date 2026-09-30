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
 */
class PardisoSolver final : public LocalSolver {
public:
    bool factorize(const core::SpMat& A) override
    {
        lu_.compute(A);
        if (lu_.info() != Eigen::Success) {
            error_ = "PARDISO factorization failed";
            return false;
        }
        return true;
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
    std::string name() const override { return "pardiso"; }

private:
    mutable Eigen::PardisoLU<core::SpMat> lu_; ///< Factorization (mutable: iparm is changed for A^T solves).
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