/**
 * @file local_eigensolver.cpp
 * @brief Implementation of LocalEigensolver (kernel handling + Spectra / dense eigensolvers).
 */
#include "local_eigensolver.hpp"

#include "local_solver.hpp"
#include "timer.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <Spectra/GenEigsSolver.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>

namespace schwarz2lvl::prec {

using Complex = std::complex<double>;
using CVec = Eigen::VectorXcd;
using CMat = Eigen::MatrixXcd;

/// @brief Upper bound of the 2-norm: sqrt(||M||_1 ||M||_inf).
double normEstimate(const core::SpMat& M)
{
    core::Vec cs = core::Vec::Zero(M.cols()), rs = core::Vec::Zero(M.rows());
    for (Eigen::Index k = 0; k < M.outerSize(); ++k)
        for (core::SpMat::InnerIterator it(M, k); it; ++it) {
            cs[it.col()] += std::abs(it.value());
            rs[it.row()] += std::abs(it.value());
        }
    if (M.rows() == 0) return 0.0;
    return std::sqrt(cs.maxCoeff() * rs.maxCoeff());
}

/// @brief Orthonormal basis of the columns of Y (thin Householder QR, Y assumed full rank).
core::Mat orthonormalize(const core::Mat& Y)
{
    Eigen::HouseholderQR<core::Mat> qr(Y);
    return qr.householderQ() * core::Mat::Identity(Y.rows(), Y.cols());
}

/**
 * @brief Numerical kernel of a matrix M from block inverse iteration.
 * @param solve    x = (M + sigma I)^{-1} b.
 * @param M        The matrix (A~ or A~^T).
 * @param thr      Absolute threshold on ||M y|| for unit y.
 * @param probe    Initial block size.
 * @param forced_k If >= 0, return exactly the forced_k best vectors.
 * @return Orthonormal kernel basis.
 */
core::Mat detectKernel(const std::function<void(const core::Vec&, core::Vec&)>& solve, const core::SpMat& M, double thr, int probe,
                 int forced_k)
{
    const Eigen::Index n = M.rows();
    Eigen::Index p = std::min<Eigen::Index>(std::max(probe, forced_k + 2), n);
    std::mt19937 gen(20230609u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    while (true) {
        core::Mat Y(n, p);
        for (Eigen::Index j = 0; j < p; ++j)
            for (Eigen::Index i = 0; i < n; ++i) Y(i, j) = dist(gen);
        Y = orthonormalize(Y);
        core::Vec y;
        for (int it = 0; it < 3; ++it) {
            for (Eigen::Index j = 0; j < p; ++j) {
                solve(Y.col(j), y);
                Y.col(j) = y;
            }
            Y = orthonormalize(Y);
        }
        // min over span(Y) of ||M y||: SVD of M Y (singular values in decreasing order).
        const core::Mat R = M * Y;
        Eigen::JacobiSVD<core::Mat> svd(R, Eigen::ComputeThinV);
        const core::Vec& s = svd.singularValues();
        Eigen::Index k = 0;
        if (forced_k >= 0)
            k = std::min<Eigen::Index>(forced_k, p);
        else
            for (Eigen::Index j = 0; j < s.size(); ++j) k += (s[j] <= thr) ? 1 : 0;
        if (forced_k < 0 && k == p && p < n) { // every probe vector is in the kernel: enlarge
            p = std::min<Eigen::Index>(2 * p, n);
            continue;
        }
        return Y * svd.matrixV().rightCols(k);
    }
}

/**
 * @brief (L ∩ K)^⊥_K = Q * {right singular vectors of B Q with nonzero singular values}.
 * @param B   D A D.
 * @param Q   Orthonormal basis of K.
 * @param thr Absolute threshold on the singular values of B Q.
 */
core::Mat kernelComplement(const core::SpMat& B, const core::Mat& Q, double thr)
{
    if (Q.cols() == 0) return core::Mat(Q.rows(), 0);
    const core::Mat BQ = B * Q;
    Eigen::JacobiSVD<core::Mat> svd(BQ, Eigen::ComputeFullV);
    const core::Vec& s = svd.singularValues();
    Eigen::Index r = 0;
    while (r < s.size() && s[r] > thr) ++r;
    return Q * svd.matrixV().leftCols(r);
}

/**
 * @brief Selects the eigenvectors with |lambda| >= 1/tau among the first @p nmax
 *        (sorted by decreasing modulus) and appends real basis vectors to @p out.
 * @return Number of columns appended.
 */
int selectEigenvectors(const CVec& lam, const CMat& U, int nmax, double tau, std::vector<core::Vec>& out,
                       std::vector<Complex>& kept)
{
    const int m = std::min<int>(nmax, static_cast<int>(lam.size()));
    const double thr = 1.0 / tau;
    int added = 0;
    for (int j = 0; j < m; ++j) {
        const Complex l = lam[j];
        kept.push_back(l);
        if (std::abs(l) < thr) continue;
        const bool real = std::abs(l.imag()) <= 1e-10 * std::abs(l);
        if (real) {
            out.push_back(U.col(j).real());
            ++added;
            continue;
        }
        // Complex pair: Re(u) and Im(u) span the real invariant subspace of (l, conj(l)).
        if (l.imag() < 0.0) {
            bool partner = false;
            for (int q = 0; q < m && !partner; ++q)
                partner = (q != j) && std::abs(lam[q] - std::conj(l)) <= 1e-8 * std::abs(l);
            if (partner) continue;
        }
        out.push_back(U.col(j).real());
        out.push_back(U.col(j).imag());
        added += 2;
    }
    return added;
}

/// @brief Assembles Z = [kernel part | eigenvectors] (each eigenvector normalized).
core::Mat assembleZ(const core::Mat& Kpart, const std::vector<core::Vec>& ev)
{
    core::Mat Z(Kpart.rows(), Kpart.cols() + static_cast<Eigen::Index>(ev.size()));
    Z.leftCols(Kpart.cols()) = Kpart;
    for (std::size_t j = 0; j < ev.size(); ++j) {
        const double nrm = ev[j].norm();
        Z.col(Kpart.cols() + static_cast<Eigen::Index>(j)) = (nrm > 0.0) ? core::Vec(ev[j] / nrm) : ev[j];
    }
    return Z;
}

/**
 * @class ProjectedOperator
 * @brief Matrix-free projected operator T = Pi_K (A~ + sigma I)^{-1} Pi_R B Pi_K
 *        in the form required by Spectra (Scalar, rows, cols, perform_op).
 */
class ProjectedOperator {
public:
    using Scalar = double; ///< Required by Spectra.

    ProjectedOperator(const core::SpMat& B, const linalg::LocalSolver& F, const core::Mat& Q, const core::Mat& P)
        : B_(B), F_(F), Q_(Q), P_(P), n_(B.rows()) {}
    Eigen::Index rows() const { return n_; } ///< Operator size.
    Eigen::Index cols() const { return n_; } ///< Operator size.

    /// @brief y = T x.
    void perform_op(const double* x_in, double* y_out) const
    {
        Eigen::Map<const core::Vec> x(x_in, n_);
        Eigen::Map<core::Vec> y(y_out, n_);
        v_ = x;
        if (Q_.cols() > 0) v_.noalias() -= Q_ * (Q_.transpose() * x);    // Pi_K x
        w_.noalias() = B_ * v_;                                            // B Pi_K x
        if (P_.cols() > 0) w_.noalias() -= P_ * (P_.transpose() * w_);   // Pi_R: onto range(A~)
        F_.solve(w_, z_);                                                  // (A~ + sigma I)^{-1}
        y = z_;
        if (Q_.cols() > 0) y.noalias() -= Q_ * (Q_.transpose() * z_);    // Pi_K
        ++count_;
    }
    long count() const { return count_; } ///< Number of applications.

private:
    const core::SpMat& B_;
    const linalg::LocalSolver& F_;
    const core::Mat& Q_;
    const core::Mat& P_;
    Eigen::Index n_;
    mutable core::Vec v_, w_, z_;
    mutable long count_ = 0;
};

LocalEigensolverResult LocalEigensolver::compute(const core::SpMat& Aii, const core::SpMat& Atilde, const core::Vec& D,
                                                 bool force_dense) const
{
    utils::ScopedTimer timer("setup.eigensolver");
    // B_i = D_i A_ii D_i.
    const core::SpMat B = D.asDiagonal() * Aii * D.asDiagonal();
    const Eigen::Index n = Aii.rows();
    const int nev = std::max(0, opt_.nev);
    const int ncv = (opt_.ncv > 0) ? opt_.ncv : std::max(2 * nev + 1, nev + 20);
    if (force_dense || n <= opt_.dense_threshold || ncv >= n) return computeDense(Atilde, B);
    return computeSparse(Atilde, B);
}

LocalEigensolverResult LocalEigensolver::computeDense(const core::SpMat& Atilde, const core::SpMat& B) const
{
    LocalEigensolverResult res;
    res.dense = true;
    const Eigen::Index n = Atilde.rows();
    const core::Mat At = core::Mat(Atilde);
    const core::Mat Bd = core::Mat(B);

    // SVD of A~: numerical rank, kernels and pseudo-inverse.
    Eigen::JacobiSVD<core::Mat> svd(At, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const core::Vec& s = svd.singularValues();
    const double thr = opt_.kernel_tol * (s.size() ? s[0] : 0.0);
    Eigen::Index r = 0;
    while (r < s.size() && s[r] > thr) ++r;
    const core::Mat Q = svd.matrixV().rightCols(n - r);
    res.kernel_dim = static_cast<int>(n - r);
    const core::Mat Apinv = svd.matrixV().leftCols(r) * s.head(r).cwiseInverse().asDiagonal() *
                      svd.matrixU().leftCols(r).transpose();

    const core::Mat Kpart = kernelComplement(B, Q, opt_.kernel_tol * normEstimate(B));
    res.kernel_part = static_cast<int>(Kpart.cols());

    std::vector<core::Vec> ev;
    const int nev_reg = std::max(0, opt_.nev - res.kernel_part);
    if (nev_reg > 0) {
        // T = A~^+ Pi_R B Pi_K = A~^+ B Pi_K (A~^+ already annihilates range(A~)^⊥).
        core::Mat T = Apinv * Bd;
        if (Q.cols() > 0) T -= (T * Q) * Q.transpose();
        Eigen::EigenSolver<core::Mat> es(T, true);
        if (es.info() != Eigen::Success) throw std::runtime_error("dense eigensolver failed");
        CVec lam = es.eigenvalues();
        CMat U = es.eigenvectors();
        std::vector<Eigen::Index> order(static_cast<std::size_t>(lam.size()));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](Eigen::Index a, Eigen::Index b) { return std::abs(lam[a]) > std::abs(lam[b]); });
        CVec lamS(lam.size());
        CMat US(U.rows(), U.cols());
        for (std::size_t k = 0; k < order.size(); ++k) {
            lamS[static_cast<Eigen::Index>(k)] = lam[order[k]];
            US.col(static_cast<Eigen::Index>(k)) = U.col(order[k]);
        }
        res.n_computed = std::min<int>(nev_reg, static_cast<int>(lamS.size()));
        res.n_selected = selectEigenvectors(lamS, US, nev_reg, opt_.tau, ev, res.eigenvalues);
    }
    res.Z = assembleZ(Kpart, ev);
    return res;
}

LocalEigensolverResult LocalEigensolver::computeSparse(const core::SpMat& Atilde, const core::SpMat& B) const
{
    LocalEigensolverResult res;
    const Eigen::Index n = Atilde.rows();
    const double normA = normEstimate(Atilde);

    // 1. Factorization of A~ + sigma I (retry with a larger shift if it fails).
    auto F = linalg::makeLocalSolver(opt_.local_solver);
    res.backend = F->name();
    core::SpMat Id(n, n);
    Id.setIdentity();
    double sigma = opt_.shift_rel * (normA > 0.0 ? normA : 1.0);
    core::SpMat As;
    {
        utils::ScopedTimer t("setup.eigensolver.factorization");
        bool ok = false;
        for (int attempt = 0; attempt < 4 && !ok; ++attempt) {
            As = Atilde + sigma * Id;
            As.makeCompressed();
            ok = F->factorize(As);
            if (!ok) sigma *= 1e3;
        }
        if (!ok) throw std::runtime_error("cannot factorize A~_ii + sigma I: " + F->lastError());
    }
    res.shift = sigma;

    // 2. Right kernel Q and, for nonsymmetric matrices, left kernel P.
    core::Mat Q, P;
    {
        utils::ScopedTimer t("setup.eigensolver.kernel");
        const double thr = opt_.kernel_tol * normA;
        Q = detectKernel([&](const core::Vec& b, core::Vec& x) { F->solve(b, x); }, Atilde, thr, opt_.kernel_probe, -1);
        res.kernel_dim = static_cast<int>(Q.cols());
        if (opt_.symmetric || Q.cols() == 0) {
            P = Q;
        } else {
            const core::SpMat AtT = Atilde.transpose();
            if (F->hasTransposeSolve()) {
                P = detectKernel([&](const core::Vec& b, core::Vec& x) { F->solveTranspose(b, x); }, AtT, thr, opt_.kernel_probe,
                                 static_cast<int>(Q.cols()));
            } else {
                auto FT = linalg::makeLocalSolver("sparselu");
                core::SpMat AsT = As.transpose();
                if (!FT->factorize(AsT)) throw std::runtime_error("cannot factorize (A~_ii + sigma I)^T");
                P = detectKernel([&](const core::Vec& b, core::Vec& x) { FT->solve(b, x); }, AtT, thr, opt_.kernel_probe,
                                 static_cast<int>(Q.cols()));
            }
        }
    }

    // 3. Infinite eigenvalues: (L ∩ K)^⊥_K.
    const core::Mat Kpart = kernelComplement(B, Q, opt_.kernel_tol * normEstimate(B));
    res.kernel_part = static_cast<int>(Kpart.cols());

    // 4. Finite eigenvalues of T = Pi_K (A~ + sigma I)^{-1} Pi_R B Pi_K with Spectra.
    std::vector<core::Vec> ev;
    int nev_reg = std::max(0, opt_.nev - res.kernel_part);
    nev_reg = std::min<int>(nev_reg, static_cast<int>(n) - 2);
    if (nev_reg > 0) {
        utils::ScopedTimer t("setup.eigensolver.arnoldi");
        int ncv = (opt_.ncv > 0) ? opt_.ncv : std::max(2 * nev_reg + 1, nev_reg + 20);
        ncv = std::min<int>(std::max(ncv, nev_reg + 2), static_cast<int>(n));
        ProjectedOperator op(B, *F, Q, P);
        Spectra::GenEigsSolver<ProjectedOperator> eigs(op, nev_reg, ncv);
        // Deterministic starting vector in K^⊥.
        core::Vec v0 = core::Vec::Ones(n) + 0.1 * core::Vec::LinSpaced(n, -1.0, 1.0).array().sin().matrix();
        if (Q.cols() > 0) v0 -= Q * (Q.transpose() * v0);
        eigs.init(v0.data());
        eigs.compute(Spectra::SortRule::LargestMagn, opt_.maxit, opt_.tol, Spectra::SortRule::LargestMagn);
        res.converged = (eigs.info() == Spectra::CompInfo::Successful);
        const CVec lam = eigs.eigenvalues();   // converged eigenvalues, decreasing modulus
        const CMat U = eigs.eigenvectors();
        res.n_computed = static_cast<int>(lam.size());
        res.matvecs = op.count();
        res.n_selected = selectEigenvectors(lam, U, nev_reg, opt_.tau, ev, res.eigenvalues);
    }
    res.Z = assembleZ(Kpart, ev);
    return res;
}

} // namespace schwarz2lvl::prec
