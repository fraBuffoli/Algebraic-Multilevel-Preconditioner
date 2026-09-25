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
    std::string matrix_file;                ///< Matrix Marix Market file name (read on rank 0).
    std::string rhs = "random";             ///< "random", "ones", "generated" or a Matrix Market array file.
    std::string generate;                   ///< Built-in generator (laplace2d, laplace3d, convdiff2d, convdiff3d, coupled2d, coupled3d).
    int nx = 64;                            ///< Generator: cells along x-axis.
    int ny = 0;                             ///< Generator: cells along y-axis (0 = same as nx).
    int nz = 0;                             ///< Generator: cells along z-axis (0 = same as nx).
    double nu = 1.0;                        ///< Generator: viscosity.
    int ncomp = 4;                          ///< Generator: numeber of unknowns per cell for coupled problems. 
    int block_size = 0;                     ///< Unknowns per cell (0 = automatic: 1 or ncomp).     
    unsigned seed = 12345;                  ///< Seed of the random right-hand side. 

    // PARTITIONING PARAMETERS
    std::string metis_objective = "cut";    ///< METIS objective: "cut" or "vol".
    bool metis_vertex_weights = false;      ///< METIS: Weight vertices by their number of nonzeros.

    // PRECONDITIONER PARAMETERS
    std::string pou = "boolean";            ///< Partition of unity: "boolean" or "multiplicity".
    std::string one_level = "ras";          ///< One level method: "ras" or "asm".
    std::string coarse = "deflated";        ///< Coarse correction: "deflated", "additive" or none.
    double tau = 0.6;                       ///< Threshold: select |lambda| >= 1/tau.
    int nev = 300;                          ///< Maximum number of eigenvectors per subdomain for the coarse space.
    double eig_tol = 1e-8;                  ///< Relative tolerance of the Krylov eigensolver.
    int eig_maxit = 1000;                   ///< Maximum number of restarts of the eigensolver.
    int eig_ncv = 0;                        ///< Krylov subspace size (0 = automatic).
    double shift_rel = 1e-13;               ///< Relative shift sigma/||A~_ii|| used to factorize A~_ii.
    double kernel_tol = 1e-8;               ///< Relative threshold defining the numerical kernel of A~_ii.
    int kernel_probe = 8;                   ///< Initial block size of the kernel detection.
    int dense_threshold = 400;              ///< Subdomains with n_i <= threshold use dense eigensolvers.
    double orth_tol = 1e-10;                ///< Rank threshold of the local coarse basis orthonormalization.
    std::string local_solver = "auto";      ///< Local LU backend: "auto", "sparselu".
    std::string coarse_solver = "auto";     ///< Coarse solver: "auto", "mumps" (distributed) or "root".
    int coarse_procs = 0;                   ///< Processes used for the distributed coarse factorization (0 = all).

    // KRYLOV SOLVER PARAMETERS
    int restart = 30;                       ///< GMRES restart length.
    double rtol = 1e-8;                     ///< Relative tolerance on the unpreconditioned residual.
    int maxit = 1000;                       ///< Maximum number of GMRES iterations.

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