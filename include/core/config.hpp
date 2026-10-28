/**
* @file config.hpp
* @brief Run-time configuration of the solver driver and command-line parameters.
*/

#ifndef SCHWARZ2LVL_CORE_CONFIG_HPP
#define SCHWARZ2LVL_CORE_CONFIG_HPP    

#include <string>
#include <iosfwd>

namespace schwarz2lvl::core{
/**
 * @brief All user-selectable parameters.
 * 
 * Default values reproduce the setup of the numerical experiments in the reference
 * paper. Zero initial guess, random right-hand side.
 */
struct SolverConfig{
    // INPUT PARAMETERS
    std::string matrix_file;                ///< Matrix Market file name (read on rank 0).
    std::string rhs = "random";             ///< "random", "ones", "generated" or a Matrix Market array file.
    std::string generate;                   ///< Built-in generator (laplace2d, laplace3d, convdiff2d, convdiff3d, coupled2d, coupled3d).
    int nx = 64;                            ///< Generator: cells along x-axis.
    int ny = 0;                             ///< Generator: cells along y-axis (0 = same as nx).
    int nz = 0;                             ///< Generator: cells along z-axis (0 = same as nx).
    double nu = 1.0;                        ///< Generator: viscosity.
    int ncomp = 4;                          ///< Generator: number of unknowns per cell for coupled problems.
    int block_size = 0;                     ///< Unknowns per cell (0 = automatic: 1 or ncomp).
    unsigned seed = 12345;                  ///< Seed of the random right-hand side.
    double rho_ratio = 1000.0;              ///< Generator bubbles: liquid / gas density ratio.
    int nbubbles = 50;                      ///< Generator bubbles: number of bubbles.
    double radius = 0.06;                   ///< Generator bubbles: radius (unit domain).
    double cfl = 0.5;                       ///< Generator bubbles: rise per step [cells].
    std::string bubble_source = "alpha";    ///< Generator bubbles: --rhs generated is "alpha" (vol x alpha) or
                                            ///< "buoyancy" (p_rgh source of a VoF solver, fluid at rest).
    double pulse = 0.0;                     ///< Generators convdiff/coupled: amplitude of the convection pulsation in time.
    int pulse_period = 20;                  ///< Period of the pulsation [steps].

    // SEQUENCE OF SOLVES (time loop, same sparsity pattern)
    int steps = 0;                          ///< Steps of the time loop (0 = a single solve, as in the paper).
    int correctors = 3;                     ///< Solves per step with the same matrix and a new right-hand side.
    std::string reuse = "auto";             ///< Setup reuse between solves: "off" (rebuild every solve), "refactor" (never rebuild) or "auto" (break-even).
    double rebuild_growth = 4.0;            ///< auto: also rebuild when iterations per decade exceed this factor times the reference (0 = off).
    int rebuild_every = 0;                  ///< auto: also rebuild after this many refactorizations (0 = off).
    std::string steps_csv;                  ///< Append one line per solve to this CSV file (rank 0).
    bool warm_start = false;                ///< Every solve starts from the solution of the previous one (else from 0).
    double final_rtol = -1.0;               ///< rtol of the last corrector of every step (as p_rghFinal; < 0: rtol).
    double final_atol = -1.0;               ///< atol of the last corrector of every step (< 0: atol).

    // PARTITIONING PARAMETERS
    std::string metis_objective = "cut";    ///< METIS objective: "cut" or "vol".
    bool metis_vertex_weights = false;      ///< METIS: Weight vertices by their number of nonzeros.

    // PRECONDITIONER PARAMETERS
    std::string pou = "boolean";            ///< Partition of unity: "boolean" or "multiplicity".
    std::string one_level = "ras";          ///< One level method: "ras" or "asm".
    std::string coarse = "deflated";        ///< Coarse correction: "deflated", "additive" or none.
    double tau = 0.6;                       ///< Threshold: select |lambda| >= 1/tau.
    int nev = 300;                          ///< Maximum number of eigenvectors per subdomain for the coarse space.
    int nev_start = 32;                     ///< First number of eigenpairs computed, doubled until |lambda| < 1/tau (0 = nev).
    double eig_tol = 1e-8;                  ///< Relative tolerance of the Krylov eigensolver.
    int eig_maxit = 1000;                   ///< Maximum number of restarts of the eigensolver.
    int eig_ncv = 0;                        ///< Krylov subspace size (0 = automatic).
    double shift_rel = 1e-13;               ///< Relative shift sigma/||A~_ii|| used to factorize A~_ii.
    double kernel_tol = 1e-8;               ///< Relative threshold defining the numerical kernel of A~_ii.
    int kernel_probe = 8;                   ///< Initial block size of the kernel detection.
    int dense_threshold = 400;              ///< Subdomains with n_i <= threshold use dense eigensolvers.
    double orth_tol = 1e-10;                ///< Rank threshold of the local coarse basis orthonormalization.
    std::string local_solver = "auto";      ///< Local backend: "auto", "sparselu", "umfpack", "pardiso".
    std::string local_factorization = "auto"; ///< "auto" (symmetric A: Cholesky -> LDL^T -> LU), "lu", "cholesky", "ldlt".
    std::string local_precision = "double"; ///< Precision of the factors of A_ii (one-level part): "double" or "single".
    std::string local_approx = "none";      ///< A_ii^{-1} of the one-level part (P7): "none" (exact factors) or "amg".
    std::string amg_coarsening = "sa";      ///< --local-approx amg: "sa" (smoothed aggregation) or "rs" (Ruge-Stuben).
    int amg_sweeps = 1;                     ///< --local-approx amg: Gauss-Seidel steps before/after the coarse correction.
    int amg_coarse_enough = 500;            ///< --local-approx amg: maximum size of the coarsest level (skyline LU).
    bool amg_reuse = false;                 ///< --local-approx amg: refactorization keeps the transfer operators.
    std::string coarse_solver = "auto";     ///< Coarse solver: "auto", "mumps", "cpardiso" (distributed) or "root".
    std::string coarse_factorization = "auto"; ///< A_00 (root, cpardiso): "auto" (symmetric A: Cholesky -> LDL^T [-> LU]), "lu", "cholesky", "ldlt".
    int coarse_procs = 0;                   ///< Processes used for the distributed coarse factorization (0 = all).
    long long coarse_root_below = 0;        ///< n0 below this: coarse solver "root" whatever coarse_solver says (0 = off).

    // KRYLOV SOLVER PARAMETERS
    std::string krylov = "gmres";           ///< Krylov method: "gmres" (restarted, right preconditioning) or "bicgstab"
                                            ///< (right preconditioning, the algorithm of OpenFOAM's PBiCGStab).
    int restart = 30;                       ///< GMRES restart length.
    double rtol = 1e-8;                     ///< Relative tolerance on the unpreconditioned residual.
    double atol = 0.0;                      ///< --stop-norm openfoam only: absolute tolerance on the normalised
                                            ///< residual (OpenFOAM `tolerance`): stop when sum|r| / F <= atol.
    std::string stop_norm = "l2";           ///< Stopping test: "l2" (||r||_2 <= rtol ||b||_2) or "openfoam"
                                            ///< (OpenFOAM normalised residual, relTol semantics: sum|r| <= rtol sum|r0|).
    int maxit = 1000;                       ///< Maximum number of iterations (applications of the preconditioner).

    // OUTPUT PARAMETERS
    int verbose = 1;                        ///< 0 = quiet, 1 = summary, 2 = per-rank details.
    std::string csv;                        ///< Append a result line to this CSV file (rank 0).
    std::string history;                    ///< Write the residual history to this file (rank 0).
    std::string write_solution;             ///< Write the solution (original ordering) to this file.

    /**
     * @brief Parses the command line (`--key value` or `--key=value`).
     * @param argc Argument count.
     * @param argv Argument vector.
     * @return The parsed configuration.
     * @throws std::invalid_argument on unknown keys or malformed values.
     */
    static SolverConfig fromCommandLine(int argc, char** argv);

    /// @brief Prints the configuration.
    void print(std::ostream& os) const;

    /// @brief Returns the help message.
    static std::string help();

    /// @brief Checks the consistency of the parameters.
    void validate() const;
};

} // namespace schwarz2lvl::core

 #endif // SCHWARZ2LVL_CORE_CONFIG_HPP