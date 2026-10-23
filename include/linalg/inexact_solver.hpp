/**
 * @file inexact_solver.hpp
 * @brief Approximate inverse of the subdomain matrices A_ii for the one-level part (P7): AMG.
 *
 * The paper factorizes every A_ii exactly. With large subdomains the solves with those factors
 * are the largest cost of a GMRES iteration (P5: 60-70% at ~30k cells per process), and the
 * Cholesky factor of a 3D subdomain has 20-35 times the nonzeros of A_ii (PARDISO, METIS ordering,
 * 7-point Laplacian of 30k / 69k unknowns; fill-in ~ n^(4/3)).
 * Schwarz theory (Toselli-Widlund, Ch. 2, local stability constant omega; Dolean-Jolivet-Nataf,
 * end of Sect. 6.3) allows any local solver B_i ~ A_ii^{-1} that is *spectrally equivalent* to
 * A_ii^{-1} (c1 <= x^T A_ii B_i A_ii x / x^T A_ii x <= c2 with constants independent of the
 * subdomain size): the convergence is then asymptotically that of the exact local solves, with a
 * condition number larger by a factor of order c2 / c1.
 *
 * B_i = one AMG V-cycle (AMGCL, header-only, MIT, external/amgcl, build with USE_AMGCL=1) on the
 * builtin (sequential) backend, Gauss-Seidel smoother, two coarsenings:
 *  - "sa" smoothed aggregation (default): cheap setup, operator complexity ~1.5;
 *  - "rs" Ruge-Stuben (classical AMG): iterations closest to the exact solves, but denser coarse
 *    levels (operator complexity ~2.7) and a more expensive setup.
 * On the G100 bubbles problem (rho ratio 1000, 240 processes) the first iterations after a rebuild
 * were those of the exact solves, the local solves 3.4x (30k cells) to 4.3x (70k cells) cheaper,
 * the cost per solve -47% / -54% with "sa". Incomplete Cholesky (not spectrally equivalent: 2.4x
 * the iterations of the exact solves at 7k cells per subdomain, 8.7x at 70k) and the spai0 and
 * Chebyshev smoothers (equal or worse than Gauss-Seidel) were tried in the first P7 campaign and
 * removed, together with the ILU(0) smoother (only tested locally).
 *
 * Every solve() starts from x = 0 and performs a fixed number of operations, so B_i is a fixed
 * linear operator and plain (not flexible) GMRES remains valid. The coarse space (eigenproblems
 * with A~_ii) and A_00 keep their exact factorizations: only the one-level part changes.
 *
 * refactorize() (same pattern, new values): by default a full AMG setup. With AmgOptions::reuse
 * the transfer operators P, R of the last setup are kept and only the Galerkin products R A P and
 * the smoothers are recomputed (AMGCL rebuild()): cheaper, but P was built on the old matrix and
 * the bubbles move, so the V-cycle loses quality step after step (G100: more iterations between
 * two rebuilds of the coarse space, no gain overall).
 * `single` precision runs the hierarchy in float (GMRES stays in double).
 */
#ifndef SCHWARZ2LVL_LINALG_INEXACT_SOLVER_HPP
#define SCHWARZ2LVL_LINALG_INEXACT_SOLVER_HPP

#include "local_solver.hpp"

namespace schwarz2lvl::linalg {

/**
 * @brief Creates the approximate inverse requested by @p spec (spec.approx = "amg").
 * @throws std::invalid_argument if the approximation or an AMG option is unknown, or "amg" is
 *         requested in a build without USE_AMGCL.
 */
std::unique_ptr<LocalSolver> makeInexactSolver(const LocalSolverSpec& spec);

/// @brief True if this build contains AMGCL (USE_AMGCL=1).
bool amgAvailable();

} // namespace schwarz2lvl::linalg

#endif // SCHWARZ2LVL_LINALG_INEXACT_SOLVER_HPP
