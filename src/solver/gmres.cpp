/**
 * @file gmres.cpp
 * @brief Implementation of Gmres.
 */
#include "gmres.hpp"

#include "timer.hpp"
#include "vector_ops.hpp"

#include <cmath>
#include <stdexcept>

namespace schwarz2lvl::solv {

StopNorm parseStopNorm(const std::string& s)
{
    if (s == "l2") return StopNorm::L2;
    if (s == "openfoam") return StopNorm::OpenFOAM;
    throw std::invalid_argument("unknown stopping norm '" + s + "' (l2|openfoam)");
}

const char* toString(StopNorm s) { return s == StopNorm::L2 ? "l2" : "openfoam"; }

namespace {

/// @brief sum_i |v_i| over all ranks.
double norm1(const core::Vec& v, MPI_Comm comm)
{
    double s = v.cwiseAbs().sum();
    MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_DOUBLE, MPI_SUM, comm);
    return s;
}

/// @brief OpenFOAM normalisation factor (lduMatrix::solver::normFactor) of the initial guess x0:
///        F = sum|A x0 - xm A 1| + sum|b - xm A 1| + 1e-20, xm = mean of x0 over all rows.
double openfoamNormFactor(const linalg::DistributedMatrix& A, const core::Vec& b, const core::Vec& x0, const core::Vec& Ax0,
                          MPI_Comm comm)
{
    double sc[2] = {x0.sum(), static_cast<double>(x0.size())};
    MPI_Allreduce(MPI_IN_PLACE, sc, 2, MPI_DOUBLE, MPI_SUM, comm);
    const double xm = (sc[1] > 0.0) ? sc[0] / sc[1] : 0.0;
    if (xm == 0.0) return norm1(Ax0, comm) + norm1(b, comm) + 1e-20;
    core::Vec rowsum;
    A.apply(core::Vec::Ones(x0.size()), rowsum);
    return norm1(Ax0 - xm * rowsum, comm) + norm1(b - xm * rowsum, comm) + 1e-20;
}

} // namespace

GmresResult Gmres::solve(const linalg::DistributedMatrix& A, const prec::Preconditioner& M, const core::Vec& b, core::Vec& x) const
{
    utils::ScopedTimer timer("solve");
    MPI_Comm comm = A.comm();
    const Eigen::Index n = b.size();
    const bool of = (stop_ == StopNorm::OpenFOAM);
    GmresResult res;
    const double bnorm = linalg::norm2(b, comm);
    if (x.size() != n) x = core::Vec::Zero(n);
    if (bnorm == 0.0) {
        x.setZero();
        res.converged = true;
        res.history.push_back(0.0);
        return res;
    }

    core::Mat V(n, m_ + 1);                 // Krylov basis (owned rows)
    core::Mat H = core::Mat::Zero(m_ + 1, m_);    // Hessenberg matrix (rotated in place)
    core::Vec cs(m_), sn(m_), g(m_ + 1);
    core::Vec r(n), w(n), z(n), rj;

    A.apply(x, r);
    double normF = 1.0, l1_0 = 0.0, l1 = 0.0; // OpenFOAM test: sum|r| <= rtol sum|r0|
    if (of) normF = openfoamNormFactor(A, b, x, r, comm);
    r = b - r;
    double beta = linalg::norm2(r, comm);
    res.history.push_back(beta / bnorm);
    res.rel_residual = beta / bnorm;
    if (of) {
        l1_0 = l1 = norm1(r, comm);
        res.initial_residual_of = l1_0 / normF;
        if (l1_0 == 0.0) res.converged = true;
    } else if (beta <= rtol_ * bnorm) {
        res.converged = true;
    }

    while (!res.converged && res.iterations < maxit_) {
        V.col(0) = r / beta;
        g.setZero();
        g[0] = beta;
        H.setZero();
        int k = 0; // columns of the current cycle
        bool breakdown = false;
        for (int j = 0; j < m_ && res.iterations < maxit_; ++j) {
            // w = A M^{-1} v_j
            {
                utils::ScopedTimer tp("solve.preconditioner");
                M.apply(V.col(j), z);
            }
            {
                utils::ScopedTimer ta("solve.spmv");
                A.apply(z, w);
            }
            double hn = 0.0;
            {
                // CGS2: two passes of classical Gram-Schmidt, one reduction each.
                utils::ScopedTimer to("solve.orthogonalization");
                core::Vec h = linalg::multiDot(V, j + 1, w, comm);
                w.noalias() -= V.leftCols(j + 1) * h;
                const core::Vec h2 = linalg::multiDot(V, j + 1, w, comm);
                w.noalias() -= V.leftCols(j + 1) * h2;
                h += h2;
                hn = linalg::norm2(w, comm);
                H.col(j).head(j + 1) = h;
                H(j + 1, j) = hn;
                if (hn > 0.0) V.col(j + 1) = w / hn;
            }
            // Apply previous rotations to the new column, then compute a new one.
            for (int i = 0; i < j; ++i) {
                const double t = cs[i] * H(i, j) + sn[i] * H(i + 1, j);
                H(i + 1, j) = -sn[i] * H(i, j) + cs[i] * H(i + 1, j);
                H(i, j) = t;
            }
            const double den = std::hypot(H(j, j), H(j + 1, j));
            cs[j] = (den > 0.0) ? H(j, j) / den : 1.0;
            sn[j] = (den > 0.0) ? H(j + 1, j) / den : 0.0;
            H(j, j) = den;
            H(j + 1, j) = 0.0;
            g[j + 1] = -sn[j] * g[j];
            g[j] = cs[j] * g[j];
            ++k;
            ++res.iterations;
            res.rel_residual = std::abs(g[j + 1]) / bnorm;
            res.history.push_back(res.rel_residual);
            bool done = false;
            if (of) {
                // GMRES residual r_j = g_{j+1} V_{j+1} Q^T e_{j+1}: one pass over the basis.
                utils::ScopedTimer ts("solve.stopping_test");
                if (hn == 0.0 || g[j + 1] == 0.0) {
                    l1 = 0.0;
                } else {
                    core::Vec q = core::Vec::Zero(j + 2);
                    q[j + 1] = 1.0;
                    for (int i = j; i >= 0; --i) { // Q^T = G_0^T ... G_j^T
                        const double a = q[i], c = q[i + 1];
                        q[i] = cs[i] * a - sn[i] * c;
                        q[i + 1] = sn[i] * a + cs[i] * c;
                    }
                    rj.noalias() = V.leftCols(j + 2) * q;
                    l1 = std::abs(g[j + 1]) * norm1(rj, comm);
                }
                done = l1 <= rtol_ * l1_0;
            } else {
                done = res.rel_residual <= rtol_;
            }
            if (done) {
                res.converged = true;
                break;
            }
            if (hn == 0.0) { // happy breakdown: exact solution in the Krylov space
                breakdown = true;
                break;
            }
        }
        // x += M^{-1} V_k y_k with H_k y_k = g_k (upper triangular after the rotations).
        const core::Vec y = H.topLeftCorner(k, k).triangularView<Eigen::Upper>().solve(g.head(k));
        const core::Vec u = V.leftCols(k) * y;
        {
            utils::ScopedTimer tp("solve.preconditioner");
            M.apply(u, z);
        }
        x += z;
        // True residual for the next cycle.
        A.apply(x, r);
        r = b - r;
        beta = linalg::norm2(r, comm);
        if (of) l1 = norm1(r, comm);
        const bool within = of ? (l1 <= rtol_ * l1_0) : (beta <= rtol_ * bnorm);
        const bool within_slack = of ? (l1 <= 1.1 * rtol_ * l1_0) : (beta <= 1.1 * rtol_ * bnorm);
        if (breakdown) res.converged = within || res.converged;
        // The GMRES estimate can be optimistic when M^{-1} is not exactly linear (e.g. factors in
        // single precision): accept convergence only if the true residual agrees (10% slack: in
        // double precision both agree to ~1e-15, so this never triggers there). Otherwise restart
        // from the true residual, which is mixed-precision iterative refinement.
        if (res.converged && !within_slack) res.converged = false;
        if (beta == 0.0) res.converged = true;
    }
    res.true_rel_residual = beta / bnorm;
    if (of) res.final_residual_of = l1 / normF;
    return res;
}

} // namespace schwarz2lvl::solv
