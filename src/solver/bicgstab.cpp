/**
 * @file bicgstab.cpp
 * @brief Implementation of BiCgStab.
 */
#include "bicgstab.hpp"

#include "timer.hpp"
#include "vector_ops.hpp"

#include <cmath>

namespace schwarz2lvl::solv {

KrylovResult BiCgStab::solve(const linalg::DistributedMatrix& A, const prec::Preconditioner& M, const core::Vec& b,
                             core::Vec& x) const
{
    utils::ScopedTimer timer("solve");
    MPI_Comm comm = A.comm();
    const Eigen::Index n = b.size();
    const bool of = (stop_ == StopNorm::OpenFOAM);
    KrylovResult res;
    const double bnorm = linalg::norm2(b, comm);
    if (x.size() != n) x = core::Vec::Zero(n);
    if (bnorm == 0.0) {
        x.setZero();
        res.converged = true;
        res.history.push_back(0.0);
        return res;
    }

    core::Vec r(n), rh(n), p = core::Vec::Zero(n), v = core::Vec::Zero(n), y(n), s(n), z(n), t(n);
    A.apply(x, r);
    double normF = 1.0, l1_0 = 0.0; // OpenFOAM test: sum|r| <= rtol sum|r0|
    if (of) normF = detail::openfoamNormFactor(A, b, x, r, comm);
    r = b - r;

    // Global sums of k local values in one reduction.
    auto allreduce = [comm](double* d, int k) { MPI_Allreduce(MPI_IN_PLACE, d, k, MPI_DOUBLE, MPI_SUM, comm); };
    double rn = 0.0, l1 = 0.0; // ||w||_2 and sum|w| of the last residual tested
    // Stopping norms of w; with extra != nullptr also (extra, w) in the same reduction.
    auto residualNorms = [&](const core::Vec& w, const core::Vec* extra = nullptr) {
        utils::ScopedTimer ts("solve.stopping_test");
        double d[3] = {w.squaredNorm(), of ? w.cwiseAbs().sum() : 0.0, extra ? extra->dot(w) : 0.0};
        allreduce(d, 3);
        rn = std::sqrt(d[0]);
        l1 = d[1];
        return d[2];
    };
    auto satisfied = [&](double slack) { return of ? (l1 <= slack * rtol_ * l1_0) : (rn <= slack * rtol_ * bnorm); };
    auto applyM = [&](const core::Vec& in, core::Vec& out) {
        utils::ScopedTimer tp("solve.preconditioner");
        M.apply(in, out);
    };
    auto applyA = [&](const core::Vec& in, core::Vec& out) {
        utils::ScopedTimer ta("solve.spmv");
        A.apply(in, out);
    };

    residualNorms(r);
    if (of) {
        l1_0 = l1;
        res.initial_residual_of = l1_0 / normF;
    }
    res.history.push_back(rn / bnorm);
    res.rel_residual = rn / bnorm;

    // Cycles: the first one from x0, the next ones (rare) from the true residual of the current x.
    for (;;) {
        if (satisfied(1.0)) {
            res.converged = true;
            break;
        }
        if (res.iterations >= maxit_) break;
        rh = r; // shadow residual
        const double rhn = rn;
        double rho = rn * rn, rho_old = 1.0, alpha = 1.0, omega = 1.0;
        bool first = true, candidate = false;
        while (res.iterations < maxit_) {
            {
                utils::ScopedTimer tv("solve.vector_ops");
                if (first) p = r;
                else p = r + ((rho / rho_old) * (alpha / omega)) * (p - omega * v);
            }
            first = false;
            applyM(p, y);
            applyA(y, v);
            ++res.iterations;
            double rhv = 0.0;
            {
                utils::ScopedTimer tv("solve.vector_ops");
                rhv = rh.dot(v);
                allreduce(&rhv, 1);
            }
            if (rhv == 0.0 || !std::isfinite(rhv)) {
                res.breakdown = true;
                break;
            }
            alpha = rho / rhv;
            {
                utils::ScopedTimer tv("solve.vector_ops");
                s = r - alpha * v;
            }
            // Half step: s is the residual of x + alpha y.
            residualNorms(s);
            res.rel_residual = rn / bnorm;
            res.history.push_back(res.rel_residual);
            if (satisfied(1.0) || res.iterations >= maxit_) {
                utils::ScopedTimer tv("solve.vector_ops");
                x.noalias() += alpha * y;
                r.swap(s);
                candidate = satisfied(1.0);
                break;
            }
            applyM(s, z);
            applyA(z, t);
            ++res.iterations;
            {
                utils::ScopedTimer tv("solve.vector_ops");
                double d[2] = {t.squaredNorm(), t.dot(s)};
                allreduce(d, 2);
                omega = (d[0] > 0.0) ? d[1] / d[0] : 0.0;
                x.noalias() += alpha * y + omega * z;
                r = s - omega * t;
            }
            rho_old = rho;
            rho = residualNorms(r, &rh); // stopping norms of r and (r^, r) for the next step
            res.rel_residual = rn / bnorm;
            res.history.push_back(res.rel_residual);
            if (satisfied(1.0)) {
                candidate = true;
                break;
            }
            if (omega == 0.0 || std::abs(rho) < 1e-14 * rhn * rn) break; // restart with r^ = r
        }
        // True residual of x (end of the cycle).
        A.apply(x, t);
        r = b - t;
        residualNorms(r);
        if (res.breakdown) break;
        // Accept convergence only if the true residual agrees (10% slack, as in Gmres).
        if (candidate && satisfied(1.1)) {
            res.converged = true;
            break;
        }
        if (res.iterations >= maxit_) break;
        ++res.restarts;
    }
    res.true_rel_residual = rn / bnorm;
    if (of) res.final_residual_of = l1 / normF;
    return res;
}

} // namespace schwarz2lvl::solv
