/**
 * @file bicgstab.hpp
 * @brief BiCGStab with right preconditioning on distributed vectors (the algorithm of OpenFOAM's PBiCGStab).
 */
#ifndef SCHWARZ2LVL_SOLVER_BICGSTAB_HPP
#define SCHWARZ2LVL_SOLVER_BICGSTAB_HPP

#include "krylov.hpp"

#include <string>

namespace schwarz2lvl::solv {

/**
 * @class BiCgStab
 * @brief Right-preconditioned BiCGStab (van der Vorst 1992), any nonsingular A and any preconditioner.
 *
 * Same algorithm as lduMatrix::solver PBiCGStab of OpenFOAM, so that the two-level Schwarz
 * preconditioner can later be plugged into it unchanged (an lduMatrix::preconditioner): shadow
 * residual r^ = r0, right preconditioning (y = M^{-1} p, x += alpha y + omega z), omega = (t, s)/(t, t),
 * and the stopping test applied both to the intermediate residual s (half step) and to r.
 *
 * One BiCGStab step applies M^{-1} twice (p and s) and A twice; KrylovResult::iterations counts the
 * applications of M^{-1} (two per step, one if the test is met at the half step), the same unit as
 * a GMRES iteration, so that iterations, time per iteration and the reuse policy compare directly;
 * maxit also counts applications. Per step: 4 global reductions instead of the 6 of OpenFOAM, fused
 * as (r^, A y); the stopping norm of s; (t, t) with (t, s); the stopping norm of r with (r^, r) for
 * the next step. No Krylov basis: about 12 vector operations per step whatever the iteration count
 * (GMRES: O(j) per iteration for the orthogonalization, plus one pass over the basis for the
 * OpenFOAM test). The residuals s and r are explicit, so the OpenFOAM test costs sum|.| in an
 * existing reduction (timer solve.stopping_test, which also contains (r^, r)).
 *
 * BiCGStab does not minimise the residual: it needs at least as many applications of M^{-1} as
 * unrestarted GMRES, usually more, and its residual history is not monotone.
 *
 * Robustness: restart with r^ = r when |(r^, r)| < 1e-14 ||r^|| ||r|| (shadow residual nearly
 * orthogonal to r) or omega = 0; at convergence the true residual b - A x is recomputed and accepted
 * with 10% slack (as Gmres), otherwise BiCGStab restarts from it; (r^, A y) = 0 stops with
 * breakdown = true.
 */
class BiCgStab final : public KrylovSolver {
public:
    /**
     * @param rtol  Relative tolerance.
     * @param maxit Maximum number of applications of the preconditioner.
     * @param stop  Norm of the stopping test.
     * @param atol  StopNorm::OpenFOAM: absolute tolerance on the normalised residual (0 = off).
     */
    BiCgStab(double rtol, int maxit, StopNorm stop = StopNorm::L2, double atol = 0.0)
        : rtol_(rtol), atol_(atol), maxit_(maxit), stop_(stop)
    {
    }

    KrylovResult solve(const linalg::DistributedMatrix& A, const prec::Preconditioner& M, const core::Vec& b,
                       core::Vec& x) const override;

    std::string name() const override { return "BiCGStab"; }

private:
    double rtol_;   ///< Relative tolerance.
    double atol_;   ///< Absolute tolerance on the OpenFOAM normalised residual.
    int maxit_;     ///< Maximum applications of the preconditioner.
    StopNorm stop_; ///< Norm of the stopping test.
};

} // namespace schwarz2lvl::solv

#endif // SCHWARZ2LVL_SOLVER_BICGSTAB_HPP
