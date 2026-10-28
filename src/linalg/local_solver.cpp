/**
 * @file local_solver.cpp
 * @brief Implementation of the local direct solver backends.
 */
#include "local_solver.hpp"

#include "inexact_solver.hpp"

#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

#ifdef USE_UMFPACK
#include <umfpack.h>
#endif
#ifdef USE_PARDISO
#include <Eigen/PardisoSupport>
#endif

#include <chrono>
#include <functional>
#include <stdexcept>
#include <type_traits>

namespace schwarz2lvl::linalg {

const char* toString(Factorization f)
{
    switch (f) {
    case Factorization::LU: return "lu";
    case Factorization::Cholesky: return "cholesky";
    case Factorization::LDLT: return "ldlt";
    }
    return "?";
}

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

namespace {

template <typename Real> using SpMatT = Eigen::SparseMatrix<Real, Eigen::ColMajor, core::LocalIndex>;

/// @brief Calls @p f with the matrix in precision Real (no copy in double precision).
template <typename Real, typename F> void withPrecision(const core::SpMat& A, F&& f)
{
    if constexpr (std::is_same_v<Real, double>) {
        f(A);
    } else {
        const SpMatT<Real> Ar = A.template cast<Real>();
        f(Ar);
    }
}

/// @brief x = solve(b) in precision Real (b rounded, x converted back).
template <typename Real, typename B, typename X, typename S> void solveIn(const S& solve, const B& b, X& x)
{
    if constexpr (std::is_same_v<Real, double>) {
        x = solve(b);
    } else {
        x = solve(b.template cast<Real>().eval()).template cast<double>();
    }
}

std::string precisionSuffix(bool single) { return single ? " single" : ""; }

} // namespace

// ---------------------------------------------------------------- Eigen built-in solvers
/**
 * @brief Eigen built-in backend: SparseLU (supernodal LU with partial pivoting, COLAMD ordering)
 *        or SimplicialLLT / SimplicialLDLT (AMD ordering), in double or single precision.
 */
template <typename Solver, typename Real> class EigenSolver final : public LocalSolver {
public:
    explicit EigenSolver(Factorization f) : f_(f) {}

    bool factorize(const core::SpMat& A) override
    {
        withPrecision<Real>(A, [&](const auto& M) {
            s_.analyzePattern(M);
            s_.factorize(M);
        });
        if (s_.info() != Eigen::Success) {
            error_ = name() + ": factorization failed";
            return false;
        }
        return true;
    }
    bool refactorize(const core::SpMat& A) override
    {
        fell_back_ = false;
        withPrecision<Real>(A, [&](const auto& M) { s_.factorize(M); }); // ordering / elimination tree kept
        if (s_.info() == Eigen::Success) return true;
        fell_back_ = true;
        return factorize(A);
    }
    void solve(const core::Vec& b, core::Vec& x) const override
    {
        solveIn<Real>([&](const auto& r) { return s_.solve(r).eval(); }, b, x);
    }
    void solve(const core::Mat& B, core::Mat& X) const override
    {
        solveIn<Real>([&](const auto& R) { return s_.solve(R).eval(); }, B, X);
    }
    bool hasTransposeSolve() const override { return true; }
    void solveTranspose(const core::Vec& b, core::Vec& x) const override
    {
        if (f_ != Factorization::LU) { // symmetric factorization: A^T = A
            solve(b, x);
            return;
        }
        if constexpr (std::is_same_v<Solver, Eigen::SparseLU<SpMatT<Real>, Eigen::COLAMDOrdering<core::LocalIndex>>>) {
            // SparseLU::transpose() is a non-const view in Eigen 3.4.
            auto& lu = const_cast<Solver&>(s_);
            solveIn<Real>([&](const auto& r) { return lu.transpose().solve(r).eval(); }, b, x);
        }
    }
    std::string name() const override
    {
        return std::string(f_ == Factorization::LU ? "sparselu" : "eigen") + " " + toString(f_) +
               precisionSuffix(std::is_same_v<Real, float>);
    }
    Factorization factorization() const override { return f_; }

private:
    Factorization f_;
    Solver s_;
};

#ifdef USE_UMFPACK
// ---------------------------------------------------------------------- UMFPACK
/**
 * @brief SuiteSparse UMFPACK backend (unsymmetric multifrontal LU, double precision), called
 *        directly on the compressed column arrays of the Eigen matrix (no conversion).
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
    std::string name() const override { return "umfpack lu"; }

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
 * @brief Intel MKL PARDISO backend through Eigen (one thread per MPI rank: set MKL_NUM_THREADS=1):
 *        PardisoLU (mtype 11), PardisoLLT (mtype 2), PardisoLDLT (mtype -2), double or single.
 *
 * No iterative refinement (iparm(8) = 0; Eigen sets 2): inside a preconditioner and an
 * eigensolver the backward error of the factorization is sufficient, and the two refinement
 * steps (1 SpMV + 1 solve each, always performed) made every solve 3-3.7x slower. PARDISO still
 * refines automatically (2 steps) the solves of a factorization with perturbed pivots.
 * LDL^T uses Bunch-Kaufman 1x1 and 2x2 pivots (iparm(21) = 1; Eigen sets 1x1 only).
 * A Cholesky factorization of a matrix that is not positive definite fails (error -4): the
 * chain of makeLocalSolver() then moves on to LDL^T.
 * Transposed solves: iparm(12) = 2 for LU; the symmetric factorizations solve with A.
 */
template <typename Pardiso, typename Real> class PardisoSolver final : public LocalSolver {
public:
    explicit PardisoSolver(Factorization f) : f_(f)
    {
        lu_.pardisoParameterArray()[7] = 0;                           // kept by compute(): set once
        if (f_ == Factorization::LDLT) lu_.pardisoParameterArray()[20] = 1; // Bunch-Kaufman pivoting
    }

    bool factorize(const core::SpMat& A) override
    {
        withPrecision<Real>(A, [&](const auto& M) { lu_.compute(M); }); // phase 12: analysis + factorization
        if (lu_.info() != Eigen::Success) {
            error_ = "PARDISO " + std::string(toString(f_)) + " factorization failed";
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
        withPrecision<Real>(A, [&](const auto& M) { lu_.factorize(M); });
        if (lu_.info() == Eigen::Success && perturbedPivots() <= perturbed_full_) return true;
        fell_back_ = true;
        return factorize(A);
    }
    void solve(const core::Vec& b, core::Vec& x) const override
    {
        solveIn<Real>([&](const auto& r) { return lu_.solve(r).eval(); }, b, x);
    }
    void solve(const core::Mat& B, core::Mat& X) const override // one call, nrhs = B.cols()
    {
        solveIn<Real>([&](const auto& R) { return lu_.solve(R).eval(); }, B, X);
    }
    bool hasTransposeSolve() const override { return true; }
    void solveTranspose(const core::Vec& b, core::Vec& x) const override
    {
        if (f_ == Factorization::LU) lu_.pardisoParameterArray()[11] = 2; // solve with the transposed matrix
        solve(b, x);
        lu_.pardisoParameterArray()[11] = 0;
    }
    void solveTranspose(const core::Mat& B, core::Mat& X) const override
    {
        if (f_ == Factorization::LU) lu_.pardisoParameterArray()[11] = 2;
        solve(B, X);
        lu_.pardisoParameterArray()[11] = 0;
    }
    long perturbedPivots() const override { return static_cast<long>(lu_.pardisoParameterArray()[13]); }
    std::string name() const override
    {
        return std::string("pardiso ") + toString(f_) + precisionSuffix(std::is_same_v<Real, float>);
    }
    Factorization factorization() const override { return f_; }

private:
    Factorization f_;
    mutable Pardiso lu_;      ///< Factorization (mutable: iparm is changed for A^T solves).
    long perturbed_full_ = 0; ///< Perturbed pivots of the last full factorization.
};
#endif

// ---------------------------------------------------------------- fallback chain
/**
 * @brief Tries the factorizations of a chain (e.g. Cholesky -> LDL^T -> LU) until one succeeds
 *        and keeps using it: a later factorize() or refactorize() starts from the current one
 *        and moves further down the chain only if it fails.
 */
class ChainSolver final : public LocalSolver {
public:
    using Maker = std::function<std::unique_ptr<LocalSolver>(Factorization)>;
    ChainSolver(std::vector<Factorization> chain, Maker make) : chain_(std::move(chain)), make_(std::move(make)) {}

    bool factorize(const core::SpMat& A) override
    {
        for (; level_ < chain_.size(); ++level_) {
            cur_ = make_(chain_[level_]);
            if (cur_->factorize(A)) return true;
            error_ = cur_->lastError();
        }
        level_ = chain_.size() - 1; // every candidate failed: report the last one
        return false;
    }
    bool refactorize(const core::SpMat& A) override
    {
        fell_back_ = false;
        if (!cur_) return factorize(A);
        if (cur_->refactorize(A)) {
            fell_back_ = cur_->refactorizationFellBack();
            return true;
        }
        fell_back_ = true; // the current type failed even with a full factorization: next one
        if (level_ + 1 >= chain_.size()) {
            error_ = cur_->lastError();
            return false;
        }
        ++level_;
        return factorize(A);
    }
    void solve(const core::Vec& b, core::Vec& x) const override { cur_->solve(b, x); }
    void solve(const core::Mat& B, core::Mat& X) const override { cur_->solve(B, X); }
    bool hasTransposeSolve() const override { return cur_ && cur_->hasTransposeSolve(); }
    void solveTranspose(const core::Vec& b, core::Vec& x) const override { cur_->solveTranspose(b, x); }
    void solveTranspose(const core::Mat& B, core::Mat& X) const override { cur_->solveTranspose(B, X); }
    long perturbedPivots() const override { return cur_ ? cur_->perturbedPivots() : -1; }
    std::string name() const override
    {
        if (!cur_) return "chain";
        return cur_->name() + (level_ > 0 ? std::string(" (fallback from ") + toString(chain_.front()) + ")" : "");
    }
    Factorization factorization() const override { return chain_[level_]; }
    int chainLevel() const override { return static_cast<int>(level_); }

private:
    std::vector<Factorization> chain_;
    Maker make_;
    std::size_t level_ = 0;
    std::unique_ptr<LocalSolver> cur_;
};

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

std::vector<std::string> availableLocalApprox()
{
    std::vector<std::string> v{"none"};
    if (amgAvailable()) v.push_back("amg");
    return v;
}

namespace {

/// @brief One backend with one factorization type and precision.
std::unique_ptr<LocalSolver> makeSingle(const std::string& backend, Factorization f, bool single)
{
    using Fa = Factorization;
    if (backend == "sparselu") {
        if (single) {
            if (f == Fa::Cholesky) return std::make_unique<EigenSolver<Eigen::SimplicialLLT<SpMatT<float>>, float>>(f);
            if (f == Fa::LDLT) return std::make_unique<EigenSolver<Eigen::SimplicialLDLT<SpMatT<float>>, float>>(f);
            return std::make_unique<EigenSolver<Eigen::SparseLU<SpMatT<float>, Eigen::COLAMDOrdering<core::LocalIndex>>, float>>(f);
        }
        if (f == Fa::Cholesky) return std::make_unique<EigenSolver<Eigen::SimplicialLLT<SpMatT<double>>, double>>(f);
        if (f == Fa::LDLT) return std::make_unique<EigenSolver<Eigen::SimplicialLDLT<SpMatT<double>>, double>>(f);
        return std::make_unique<EigenSolver<Eigen::SparseLU<SpMatT<double>, Eigen::COLAMDOrdering<core::LocalIndex>>, double>>(f);
    }
#ifdef USE_UMFPACK
    if (backend == "umfpack") return std::make_unique<UmfpackSolver>(); // LU, double only
#endif
#ifdef USE_PARDISO
    if (backend == "pardiso") {
        if (single) {
            if (f == Fa::Cholesky) return std::make_unique<PardisoSolver<Eigen::PardisoLLT<SpMatT<float>>, float>>(f);
            if (f == Fa::LDLT) return std::make_unique<PardisoSolver<Eigen::PardisoLDLT<SpMatT<float>>, float>>(f);
            return std::make_unique<PardisoSolver<Eigen::PardisoLU<SpMatT<float>>, float>>(f);
        }
        if (f == Fa::Cholesky) return std::make_unique<PardisoSolver<Eigen::PardisoLLT<SpMatT<double>>, double>>(f);
        if (f == Fa::LDLT) return std::make_unique<PardisoSolver<Eigen::PardisoLDLT<SpMatT<double>>, double>>(f);
        return std::make_unique<PardisoSolver<Eigen::PardisoLU<SpMatT<double>>, double>>(f);
    }
#endif
    throw std::invalid_argument("local solver '" + backend + "' is unknown or not compiled in");
}

} // namespace

std::unique_ptr<LocalSolver> makeLocalSolver(const LocalSolverSpec& spec)
{
    if (spec.approx != "none") return makeInexactSolver(spec); // P7: approximate inverse
    const std::string backend = (spec.backend == "auto") ? availableLocalSolvers().front() : spec.backend;
    const std::string& req = spec.factorization;
    if (req != "auto" && req != "lu" && req != "cholesky" && req != "ldlt")
        throw std::invalid_argument("unknown local factorization '" + req + "' (auto|lu|cholesky|ldlt)");
    using Fa = Factorization;
    std::vector<Fa> chain;
    if (!spec.symmetric || req == "lu" || backend == "umfpack") chain = {Fa::LU}; // umfpack: LU only
    else if (req == "ldlt") chain = {Fa::LDLT, Fa::LU};
    else chain = {Fa::Cholesky, Fa::LDLT, Fa::LU};                                // auto, cholesky
    const bool single = spec.single && backend != "umfpack";                       // umfpack: double only
    if (chain.size() == 1) return makeSingle(backend, chain.front(), single);
    return std::make_unique<ChainSolver>(chain, [backend, single](Fa f) { return makeSingle(backend, f, single); });
}

std::unique_ptr<LocalSolver> makeLocalSolver(const std::string& name)
{
    LocalSolverSpec s;
    s.backend = name;
    return makeLocalSolver(s);
}

double warmUpLocalSolver(const LocalSolverSpec& spec)
{
    const auto t0 = std::chrono::steady_clock::now();
    const int m = 20;
    const auto id = [m](int i, int j, int k) { return static_cast<core::LocalIndex>(i + m * (j + m * k)); };
    std::vector<core::Triplet> t;
    t.reserve(static_cast<std::size_t>(7 * m * m * m));
    for (int k = 0; k < m; ++k)
        for (int j = 0; j < m; ++j)
            for (int i = 0; i < m; ++i) {
                t.emplace_back(id(i, j, k), id(i, j, k), 6.0);
                if (i > 0) t.emplace_back(id(i, j, k), id(i - 1, j, k), -1.0);
                if (i + 1 < m) t.emplace_back(id(i, j, k), id(i + 1, j, k), -1.0);
                if (j > 0) t.emplace_back(id(i, j, k), id(i, j - 1, k), -1.0);
                if (j + 1 < m) t.emplace_back(id(i, j, k), id(i, j + 1, k), -1.0);
                if (k > 0) t.emplace_back(id(i, j, k), id(i, j, k - 1), -1.0);
                if (k + 1 < m) t.emplace_back(id(i, j, k), id(i, j, k + 1), -1.0);
            }
    core::SpMat A(m * m * m, m * m * m);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();
    LocalSolverSpec ws = spec;
    ws.symmetric = true;
    auto F = makeLocalSolver(ws);
    if (F->factorize(A)) {
        core::Vec x;
        F->solve(core::Vec::Ones(A.rows()), x);
        core::Mat X; // block solve, as in the kernel detection of the eigensolver
        F->solve(core::Mat::Ones(A.rows(), 8), X);
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace schwarz2lvl::linalg
