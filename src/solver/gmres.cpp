/**
 * @file gmres.cpp
 * @brief Implementation of Gmres.
 */
#include "gmres.hpp"

#include "timer.hpp"
#include "vector_ops.hpp"

#include <cmath>

namespace schwarz2lvl::solv {

GmresResult Gmres::solve(const linalg::DistributedMatrix& A, const prec::Preconditioner& M, const core::Vec& b, core::Vec& x) const
{
    utils::ScopedTimer timer("solve");
    MPI_Comm comm = A.comm();
    const Eigen::Index n = b.size();
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
    core::Vec r(n), w(n), z(n);

    A.apply(x, r);
    r = b - r;
    double beta = linalg::norm2(r, comm);
    res.history.push_back(beta / bnorm);
    res.rel_residual = beta / bnorm;
    if (beta <= rtol_ * bnorm) res.converged = true;

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
            // CGS2: two passes of classical Gram-Schmidt, one reduction each.
            core::Vec h = linalg::multiDot(V, j + 1, w, comm);
            w.noalias() -= V.leftCols(j + 1) * h;
            const core::Vec h2 = linalg::multiDot(V, j + 1, w, comm);
            w.noalias() -= V.leftCols(j + 1) * h2;
            h += h2;
            const double hn = linalg::norm2(w, comm);
            H.col(j).head(j + 1) = h;
            H(j + 1, j) = hn;
            if (hn > 0.0) V.col(j + 1) = w / hn;
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
            if (res.rel_residual <= rtol_) {
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
        if (breakdown) res.converged = (beta <= rtol_ * bnorm) || res.converged;
        if (beta == 0.0) res.converged = true;
    }
    res.true_rel_residual = beta / bnorm;
    return res;
}

} // namespace schwarz2lvl::solv
