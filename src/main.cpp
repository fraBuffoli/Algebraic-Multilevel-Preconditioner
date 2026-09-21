// =====================================================================
//  Two-level algebraic Schwarz preconditioner - VALIDATION HARNESS
//
//  This driver is a diagnostic replacement for the production main().
//  It does three things the production driver does not:
//
//    1. Verifies the algebraic partition of unity:  sum_i R_i^T D_i R_i = I
//       (paper, section 2). A nonzero error here invalidates both the RAS
//       operator (2.3) and the coarse space (3.2), silently.
//
//    2. Reports the decomposition geometry (subdomain sizes, overlap ratio,
//       coarse-space dimension), needed to interpret iteration counts.
//
//    3. Runs the SAME linear system with three preconditioners built from
//       the SAME decomposition - one-level RAS, two-level additive,
//       two-level deflated - and tabulates iterations / residual / time.
//       This is the comparison that tests the paper's claim; comparing
//       different values of N against each other does not, because N = 1
//       degenerates into an exact direct solve.
//
//  Usage:
//     mpirun -np <N> ./schwarz_solver [matrix.mtx] [tau] [tol] [maxit] [restart]
//  Defaults: matrices/nos5.mtx  0.6  1e-8  1000  30
// =====================================================================

#include "matrix_market_io.hpp"
#include "sparse_matrix.hpp"
#include "graph_partitioner.hpp"
#include "subdomain_topology.hpp"
#include "restriction_operator.hpp"
#include "partition_of_unity.hpp"
#include "preconditioner.hpp"
#include "one_level_preconditioner.hpp"
#include "local_block_splitting.hpp"
#include "local_eigensolver.hpp"
#include "coarse_space.hpp"
#include "additive_two_level_preconditioner.hpp"
#include "deflated_two_level_preconditioner.hpp"
#include "krylov_gmres.hpp"
#include "timer.hpp"
#include "config.hpp"

#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <cstdlib>

namespace {

// ---------------------------------------------------------------------
// Runtime configuration, parsed from argv with paper-consistent defaults.
// ---------------------------------------------------------------------
struct RunConfig {
    std::string matrix_path = "matrices/nos5.mtx";
    double      tau         = schwarz2lvl::kDefaultTau;              // 0.6
    double      tolerance   = schwarz2lvl::kDefaultGmresTolerance;   // 1e-8
    int         max_iter    = schwarz2lvl::kDefaultGmresMaxIterations;
    int         restart     = schwarz2lvl::kDefaultGmresRestart;     // 30
};

RunConfig parseArgs(int argc, char* argv[]) {
    RunConfig cfg;
    if (argc > 1) cfg.matrix_path = argv[1];
    if (argc > 2) cfg.tau         = std::atof(argv[2]);
    if (argc > 3) cfg.tolerance   = std::atof(argv[3]);
    if (argc > 4) cfg.max_iter    = std::atoi(argv[4]);
    if (argc > 5) cfg.restart     = std::atoi(argv[5]);
    return cfg;
}

// ---------------------------------------------------------------------
// Result of one preconditioned solve.
// ---------------------------------------------------------------------
struct RunResult {
    std::string label;
    bool        converged = false;
    int         iterations = 0;
    double      residual   = 1.0;
    double      true_residual = 1.0;
    double      seconds    = 0.0;
    schwarz2lvl::VectorType      solution;
};

// ---------------------------------------------------------------------
// CHECK 1 - algebraic partition of unity.
//
// Applies sum_i R_i^T D_i R_i to the vector of all ones and measures the
// deviation from 1. Every rank owns exactly one term of the sum, so the
// sum itself must come from an MPI reduction - same structure as the
// one-level preconditioner.
//
// Returns the max-norm error (identical on every rank).
// ---------------------------------------------------------------------
double checkPartitionOfUnity(const schwarz2lvl::SparseMatrixWrapper& A,
                             const schwarz2lvl::RestrictionOperator& R_i,
                             const schwarz2lvl::PartitionOfUnity& D_i) {
    const Eigen::Index n = A.rows();

    const schwarz2lvl::VectorType ones = schwarz2lvl::VectorType::Ones(n);
    schwarz2lvl::VectorType local_term = schwarz2lvl::VectorType::Zero(n);
    schwarz2lvl::VectorType restricted;

    R_i.apply(ones, restricted);
    restricted = restricted.cwiseProduct(D_i.getWeights());
    R_i.applyTranspose(restricted, local_term);

    schwarz2lvl::VectorType pou_sum = schwarz2lvl::VectorType::Zero(n);
    MPI_Allreduce(local_term.data(), pou_sum.data(), static_cast<int>(n),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return (pou_sum - ones).cwiseAbs().maxCoeff();
}

// ---------------------------------------------------------------------
// CHECK 2 - decomposition geometry.
//
// sum_i n_i / n is the overlap ratio: 1.0 means no overlap at all,
// large values mean the subdomains are mostly overlap, which is the
// regime where iteration counts stop being informative.
// ---------------------------------------------------------------------
void reportGeometry(Eigen::Index n, Eigen::Index n_i, Eigen::Index n_interior,
                    Eigen::Index coarse_dim, int my_rank, int num_ranks) {
    int local_ni  = static_cast<int>(n_i);
    int local_int = static_cast<int>(n_interior);
    int min_ni = 0, max_ni = 0, sum_ni = 0, sum_int = 0;

    MPI_Reduce(&local_ni,  &min_ni,  1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_ni,  &max_ni,  1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_ni,  &sum_ni,  1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_int, &sum_int, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank != 0) return;

    std::cout << "\n--- Decomposition geometry -------------------------------\n";
    std::cout << "  Global unknowns  n            : " << n << "\n";
    std::cout << "  Subdomains       N            : " << num_ranks << "\n";
    std::cout << "  Interior nodes   sum n_Ii     : " << sum_int
              << "   (must equal n)\n";
    std::cout << "  Extended sizes   n_i          : min " << min_ni
              << " / max " << max_ni << "\n";
    std::cout << "  Overlap ratio    sum n_i / n  : "
              << std::fixed << std::setprecision(3)
              << static_cast<double>(sum_ni) / static_cast<double>(n) << "\n";
    std::cout << "  Coarse space     dim(R_0)     : " << coarse_dim;
    if (coarse_dim > 0) {
        std::cout << "   (" << std::fixed << std::setprecision(1)
                  << 100.0 * static_cast<double>(coarse_dim) /
                     static_cast<double>(n) << "% of n)";
    }
    std::cout << std::endl;
}

// ---------------------------------------------------------------------
// Runs one preconditioner and times it. x is reset to zero every time so
// all variants start from the same initial guess.
// ---------------------------------------------------------------------
RunResult runVariant(const std::string& label,
                     const schwarz2lvl::SparseMatrixWrapper& A,
                     const schwarz2lvl::VectorType& b,
                     const schwarz2lvl::Preconditioner& prec,
                     const RunConfig& cfg) {
    RunResult res;
    res.label = label;

    schwarz2lvl::VectorType x = schwarz2lvl::VectorType::Zero(A.rows());

    schwarz2lvl::KrylovGmres solver(cfg.max_iter, cfg.tolerance, cfg.restart);
    // True residual, independent of the Givens estimate accumulated by GMRES.
    const schwarz2lvl::VectorType true_r = b - A.getMatrix() * x;
    solver.setVerbose(false);   // batch mode: no per-iteration flood

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    res.converged = solver.solve(A, b, x, prec);
    MPI_Barrier(MPI_COMM_WORLD);
    res.seconds = MPI_Wtime() - t0;

    res.iterations = solver.lastIterations();
    res.residual   = solver.lastResidual();
    res.true_residual = (b - A.getMatrix() * x).norm() / b.norm();
    res.solution      = x;
    return res;
}

// ---------------------------------------------------------------------
// Serializes the per-rank timing report so the output is readable
// instead of interleaved across ranks.
// ---------------------------------------------------------------------
void orderedTimerReport(int my_rank, int num_ranks) {
    for (int r = 0; r < num_ranks; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r == my_rank) {
            schwarz2lvl::TimerRegistry::instance().report(std::cout, my_rank);
            std::cout.flush();
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

} // anonymous namespace


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int my_rank = 0, num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    const RunConfig cfg = parseArgs(argc, argv);

    {
        schwarz2lvl::ScopedTimer total_app_timer("main.total_execution");

        schwarz2lvl::SparseMatrixWrapper A;
        std::vector<int> partition_map;
        bool setup_success = true;

        // -------------------------------------------------------------
        // 1. Rank 0 loads the matrix and partitions the graph with METIS
        // -------------------------------------------------------------
        if (my_rank == 0) {
            std::cout << "==========================================================\n";
            std::cout << "   TWO-LEVEL SCHWARZ - VALIDATION HARNESS\n";
            std::cout << "   Ranks (= subdomains N) : " << num_ranks << "\n";
            std::cout << "   Matrix                 : " << cfg.matrix_path << "\n";
            std::cout << "   tau                    : " << cfg.tau << "\n";
            std::cout << "   GMRES tol / maxit / m  : " << cfg.tolerance << " / "
                      << cfg.max_iter << " / " << cfg.restart << "\n";
            std::cout << "==========================================================\n\n";

            if (!schwarz2lvl::MatrixMarketIO::readMatrix(cfg.matrix_path, A)) {
                std::cerr << "[Rank 0] Critical Error: could not read matrix." << std::endl;
                setup_success = false;
            }

            if (setup_success) {
                try {
                    schwarz2lvl::GraphPartitioner partitioner;
                    partition_map = partitioner.computePartition(A, num_ranks);
                } catch (const std::exception& e) {
                    std::cerr << "[Rank 0] Partitioner Error: " << e.what() << std::endl;
                    setup_success = false;
                }
            }
        }

        MPI_Bcast(&setup_success, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        if (!setup_success) {
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        // -------------------------------------------------------------
        // 2. Broadcast matrix + partition map (fully replicated layout)
        // -------------------------------------------------------------
        int num_rows = 0;
        if (my_rank == 0) num_rows = static_cast<int>(A.rows());
        MPI_Bcast(&num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (my_rank != 0) {
            A.getMatrix().resize(num_rows, num_rows);
            partition_map.resize(num_rows);
        }

        MPI_Bcast(partition_map.data(), num_rows, MPI_INT, 0, MPI_COMM_WORLD);

        if (my_rank == 0) A.getMatrix().makeCompressed();
        MPI_Bcast(A.getMatrix().outerIndexPtr(), num_rows + 1, MPI_INT, 0, MPI_COMM_WORLD);

        int nnz = 0;
        if (my_rank == 0) nnz = static_cast<int>(A.getMatrix().nonZeros());
        MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (my_rank != 0) A.getMatrix().reserve(nnz);
        MPI_Bcast(A.getMatrix().innerIndexPtr(), nnz, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(A.getMatrix().valuePtr(), nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (my_rank != 0) A.getMatrix().makeCompressed();

        // -------------------------------------------------------------
        // 3. Local topology, restriction operator, partition of unity
        // -------------------------------------------------------------
        schwarz2lvl::SubdomainTopology topology;
        {
            schwarz2lvl::ScopedTimer t("setup.local_topology");
            topology.computeTopology(A, partition_map, my_rank);
        }

        schwarz2lvl::RestrictionOperator R_i(topology.getGlobalIndices());

        schwarz2lvl::PartitionOfUnity D_i;
        {
            schwarz2lvl::ScopedTimer t("setup.partition_of_unity");
            D_i.computeWeights(A, partition_map, topology);
        }

        // ---- CHECK 1: partition of unity --------------------------------
        const double pou_error = checkPartitionOfUnity(A, R_i, D_i);
        if (my_rank == 0) {
            std::cout << "--- Check: algebraic partition of unity ------------------\n";
            std::cout << "  max | sum_i R_i^T D_i R_i - I |  =  "
                      << std::scientific << std::setprecision(3) << pou_error << "\n";
            if (pou_error > 1e-12) {
                std::cout << "  >> FAILED. D_i does not form a partition of unity.\n"
                          << "     Neither M_RAS (2.3) nor the coarse space (3.2) is\n"
                          << "     the operator described in the paper. Iteration\n"
                          << "     counts below are not meaningful until this is 0.\n";
            } else {
                std::cout << "  >> OK\n";
            }
            std::cout.flush();
        }

        // -------------------------------------------------------------
        // 4. One-level preconditioner (local A_ii factorization)
        // -------------------------------------------------------------
        schwarz2lvl::OneLevelPreconditioner prec_1lvl;
        {
            schwarz2lvl::ScopedTimer t("setup.one_level_factorization");
            prec_1lvl.setup(A, topology, R_i, D_i);
        }

        // -------------------------------------------------------------
        // 5. Local block splitting (lumping) + local spectral problem
        // -------------------------------------------------------------
        schwarz2lvl::MatrixType A_tilde_ii = prec_1lvl.getLocalMatrix();
        {
            schwarz2lvl::ScopedTimer t("setup.lumping");
            schwarz2lvl::LocalBlockSplitting lumper;
            lumper.applyLumping(A, A_tilde_ii, topology);
        }

        schwarz2lvl::LocalEigensolver eigensolver(cfg.tau);
        {
            schwarz2lvl::ScopedTimer t("setup.local_eigensolve");
            eigensolver.computeEigenpairs(prec_1lvl.getLocalMatrix(), A_tilde_ii, D_i);
        }

        // -------------------------------------------------------------
        // 6. Gather all local bases on every rank, assemble coarse space
        // -------------------------------------------------------------
        std::vector<Eigen::MatrixXd>                  all_Z(num_ranks);
        std::vector<schwarz2lvl::RestrictionOperator> all_R(num_ranks);
        std::vector<schwarz2lvl::PartitionOfUnity>    all_D(num_ranks);

        for (int r = 0; r < num_ranks; ++r) {
            if (my_rank == r) {
                all_Z[r] = eigensolver.getLocalBase();
                all_R[r] = R_i;
                all_D[r] = D_i;
            }

            int rows_Z = static_cast<int>(all_Z[r].rows());
            int cols_Z = static_cast<int>(all_Z[r].cols());
            MPI_Bcast(&rows_Z, 1, MPI_INT, r, MPI_COMM_WORLD);
            MPI_Bcast(&cols_Z, 1, MPI_INT, r, MPI_COMM_WORLD);
            if (my_rank != r) all_Z[r].resize(rows_Z, cols_Z);
            MPI_Bcast(all_Z[r].data(), rows_Z * cols_Z, MPI_DOUBLE, r, MPI_COMM_WORLD);

            int local_size_R = 0;
            if (my_rank == r) local_size_R = static_cast<int>(R_i.localSize());
            MPI_Bcast(&local_size_R, 1, MPI_INT, r, MPI_COMM_WORLD);

            std::vector<int> indices_R(local_size_R);
            if (my_rank == r) indices_R = R_i.getGlobalIndices();
            MPI_Bcast(indices_R.data(), local_size_R, MPI_INT, r, MPI_COMM_WORLD);
            if (my_rank != r) all_R[r] = schwarz2lvl::RestrictionOperator(indices_R);

            std::vector<double> weights_D(local_size_R);
            if (my_rank == r) {
                for (int j = 0; j < local_size_R; ++j) {
                    weights_D[j] = D_i.getWeights()(j);
                }
            }
            MPI_Bcast(weights_D.data(), local_size_R, MPI_DOUBLE, r, MPI_COMM_WORLD);
            if (my_rank != r) all_D[r].setWeights(weights_D);
        }

        schwarz2lvl::CoarseSpace coarse;
        {
            schwarz2lvl::ScopedTimer t("setup.coarse_assembly_and_factorization");
            coarse.setup(A, all_Z, all_R, all_D);
        }

        // ---- CHECK 2: geometry ------------------------------------------
        reportGeometry(A.rows(),
                       R_i.localSize(),
                       static_cast<Eigen::Index>(topology.getInteriorIndices().size()),
                       coarse.coarseSize(),
                       my_rank, num_ranks);

        // -------------------------------------------------------------
        // 7. Same system, three preconditioners, same decomposition
        // -------------------------------------------------------------
        schwarz2lvl::VectorType b = schwarz2lvl::VectorType::Ones(num_rows);

        schwarz2lvl::AdditiveTwoLevelPreconditioner prec_additive(prec_1lvl, coarse);
        schwarz2lvl::DeflatedTwoLevelPreconditioner prec_deflated(A, prec_1lvl, coarse);

        if (my_rank == 0) {
            std::cout << "\n--- Running convergence comparison -----------------------"
                      << std::endl;
        }

        std::vector<RunResult> results;
        results.push_back(runVariant("One-level RAS",      A, b, prec_1lvl,     cfg));
        results.push_back(runVariant("Two-level additive", A, b, prec_additive, cfg));
        results.push_back(runVariant("Two-level deflated", A, b, prec_deflated, cfg));

        // -------------------------------------------------------------
        // 8. Summary table
        // -------------------------------------------------------------
        if (my_rank == 0) {
            std::cout << "\n==========================================================\n";
            std::cout << "  CONVERGENCE COMPARISON   (N = " << num_ranks
                      << ",  n = " << num_rows
                      << ",  dim V_0 = " << coarse.coarseSize() << ")\n";
            std::cout << "----------------------------------------------------------\n";
            std::cout << std::left  << std::setw(22) << "  Preconditioner"
                      << std::right << std::setw(8)  << "Iters"
                      << std::setw(14) << "Final res."
                      << std::setw(12) << "Solve [s]" << "\n";
            std::cout << "----------------------------------------------------------\n";

            for (const auto& r : results) {
                std::cout << std::left  << std::setw(22) << ("  " + r.label)
                          << std::right << std::setw(8);
                if (r.converged) std::cout << r.iterations;
                else             std::cout << "---";
                std::cout << std::setw(14) << std::scientific << std::setprecision(2)
                          << r.residual
                          << std::setw(12) << std::fixed << std::setprecision(4)
                          << r.seconds << "\n";
            }
            std::cout << "==========================================================\n";

            const int it1 = results[0].iterations;
            const int it2 = results[2].iterations;
            if (results[0].converged && results[2].converged && it2 > 0) {
                std::cout << "  Coarse-space speedup (1-level / deflated): "
                          << std::fixed << std::setprecision(2)
                          << static_cast<double>(it1) / static_cast<double>(it2)
                          << "x fewer iterations\n";
            } else if (!results[0].converged) {
                std::cout << "  One-level did not converge within " << cfg.max_iter
                          << " iterations - this is the expected behaviour that the\n"
                          << "  coarse space is meant to fix.\n";
            }
            
            if (results[0].converged) {
                const double nx = results[0].solution.norm();
                const double d_add = (results[1].solution - results[0].solution).norm() / nx;
                const double d_def = (results[2].solution - results[0].solution).norm() / nx;
                std::cout << "  Cross-check |x_add - x_1lvl| / |x_1lvl| : "
                          << std::scientific << std::setprecision(2) << d_add << "\n";
                std::cout << "  Cross-check |x_def - x_1lvl| / |x_1lvl| : "
                          << d_def << "\n";
            }
            
            std::cout << std::endl;
        }
    }

    orderedTimerReport(my_rank, num_ranks);

    MPI_Finalize();
    return EXIT_SUCCESS;
}