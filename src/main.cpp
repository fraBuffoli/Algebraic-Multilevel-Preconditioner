#include "matrix_market_io.hpp"
#include "sparse_matrix.hpp"
#include "graph_partitioner.hpp"
#include "subdomain_topology.hpp"
#include "restriction_operator.hpp"
#include "partition_of_unity.hpp"
#include "one_level_preconditioner.hpp"
#include "local_block_splitting.hpp"
#include "local_eigensolver.hpp"
#include "coarse_space.hpp"
#include "additive_two_level_preconditioner.hpp"
#include "deflated_two_level_preconditioner.hpp"
#include "krylov_gmres.hpp"
#include "krylov_bicgstab.hpp"
#include "timer.hpp"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cstdlib>

int main(int argc, char* argv[]) {
    // 1. Inizializzazione dell'ambiente parallelo MPI
    MPI_Init(&argc, &argv);

    int my_rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Configurazione dei solutori a runtime (Sfrutta il polimorfismo)
    const bool use_deflated = true; // true -> Deflated, false -> Additive Standard
    const bool use_gmres    = true; // true -> GMRES,    false -> BiCGStab

    {
        // Tracciamento del tempo totale dell'applicazione via RAII ScopedTimer
        schwarz2lvl::ScopedTimer total_app_timer("main.total_execution");

        const std::string mtx_filepath = "matrices/nos5.mtx";
        schwarz2lvl::SparseMatrixWrapper A;
        std::vector<int> partition_map;
        bool setup_success = true;

        // 2. Il Rank 0 (Master) esegue il caricamento e il partizionamento centralizzato con METIS
        if (my_rank == 0) {
            std::cout << "==========================================================" << std::endl;
            std::cout << "       LAUNCHING PARALLEL TWO-LEVEL SCHWARZ SOLVER        " << std::endl;
            std::cout << "       Total MPI Processes (Ranks): " << num_ranks << std::endl;
            std::cout << "       Preconditioner Type: " << (use_deflated ? "DEFLATED" : "ADDITIVE STANDARD") << std::endl;
            std::cout << "       Krylov Solver Type:  " << (use_gmres ? "GMRES" : "BiCGStab") << std::endl;
            std::cout << "==========================================================" << std::endl;

            std::cout << "\n[Rank 0] Loading Matrix Market File..." << std::endl;
            if (!schwarz2lvl::MatrixMarketIO::readMatrix(mtx_filepath, A)) {
                std::cerr << "[Rank 0] Critical Error: Could not read matrix." << std::endl;
                setup_success = false;
            }

            if (setup_success) {
                std::cout << "[Rank 0] Computing Graph Partitioning via METIS..." << std::endl;
                try {
                    schwarz2lvl::GraphPartitioner partitioner;
                    partition_map = partitioner.computePartition(A, num_ranks);
                } catch (const std::exception& e) {
                    std::cerr << "[Rank 0] Partitioner Error: " << e.what() << std::endl;
                    setup_success = false;
                }
            }
        }

        // 3. Sincronizzazione dello stato del setup iniziale
        MPI_Bcast(&setup_success, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        if (!setup_success) {
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        // 4. Distribuzione Broadcast della matrice e della mappa di partizionamento a tutto il cluster
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

        if (my_rank != 0) {
            A.getMatrix().reserve(nnz);
        }
        MPI_Bcast(A.getMatrix().innerIndexPtr(), nnz, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(A.getMatrix().valuePtr(), nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if (my_rank != 0) {
            A.getMatrix().makeCompressed();
        }

        // 5. Setup geometrico locale parallelo su ciascun rank
        schwarz2lvl::SubdomainTopology topology;
        {
            schwarz2lvl::ScopedTimer t("setup.local_topology");
            topology.computeTopology(A, partition_map, my_rank);
        }

        schwarz2lvl::RestrictionOperator R_i(topology.getGlobalIndices());
        
        schwarz2lvl::PartitionOfUnity D_i;
        D_i.computeWeights(A, partition_map, topology);

        // 6. Setup del precondizionatore locale ad 1 livello
        schwarz2lvl::OneLevelPreconditioner prec_1lvl;
        {
            schwarz2lvl::ScopedTimer t("setup.one_level_factorization");
            prec_1lvl.setup(A, topology, R_i, D_i);
        }

        // 7. Costruzione del Secondo Livello Spettrale (GenPre / Lumping)
        schwarz2lvl::MatrixType A_tilde_ii = prec_1lvl.getLocalMatrix();
        schwarz2lvl::LocalBlockSplitting lumper;
        lumper.applyLumping(A, A_tilde_ii, topology);

        double tau = 0.6; // Soglia critica di cutoff spettrale del paper
        schwarz2lvl::LocalEigensolver eigensolver(tau);
        {
            schwarz2lvl::ScopedTimer t("setup.local_eigensolve");
            eigensolver.computeEigenpairs(prec_1lvl.getLocalMatrix(), A_tilde_ii, D_i);
        }

        // 8. Raccolta centralizzata delle basi sul Rank 0 per montare il Coarse Space globale
        std::vector<Eigen::MatrixXd> all_Z(num_ranks);
        std::vector<schwarz2lvl::RestrictionOperator> all_R(num_ranks);
        std::vector<schwarz2lvl::PartitionOfUnity> all_D(num_ranks);

        for (int r = 0; r < num_ranks; ++r) {
            if (my_rank == r) {
                all_Z[r] = eigensolver.getLocalBase();
                all_R[r] = R_i;
                all_D[r] = D_i;
            }
            int rows_Z = all_Z[r].rows();
            int cols_Z = all_Z[r].cols();
            MPI_Bcast(&rows_Z, 1, MPI_INT, r, MPI_COMM_WORLD);
            MPI_Bcast(&cols_Z, 1, MPI_INT, r, MPI_COMM_WORLD);
            if (my_rank != r) {
                all_Z[r].resize(rows_Z, cols_Z);
            }
            MPI_Bcast(all_Z[r].data(), rows_Z * cols_Z, MPI_DOUBLE, r, MPI_COMM_WORLD);
            
            int local_size_R = 0;
            if (my_rank == r) local_size_R = R_i.localSize();
            MPI_Bcast(&local_size_R, 1, MPI_INT, r, MPI_COMM_WORLD);
            std::vector<int> indices_R(local_size_R);
            if (my_rank == r) indices_R = R_i.getGlobalIndices();
            MPI_Bcast(indices_R.data(), local_size_R, MPI_INT, r, MPI_COMM_WORLD);
            if (my_rank != r) {
                all_R[r] = schwarz2lvl::RestrictionOperator(indices_R);
            }

            std::vector<double> weights_D(local_size_R);
            if (my_rank == r) {
                for(int j=0; j<local_size_R; ++j) weights_D[j] = D_i.getWeights()(j);
            }
            MPI_Bcast(weights_D.data(), local_size_R, MPI_DOUBLE, r, MPI_COMM_WORLD);
            if (my_rank != r) {
                all_D[r] = all_D[my_rank]; 
            }
        }

        schwarz2lvl::CoarseSpace coarse;
        if (my_rank == 0) {
            schwarz2lvl::ScopedTimer t("setup.coarse_assembly_and_factorization");
            coarse.setup(A, all_Z, all_R, all_D);
        }

        // 9. Istanziazione polimorfica del precondizionatore a due livelli tramite l'interfaccia base
        std::unique_ptr<schwarz2lvl::Preconditioner> preconditioner;
        if (use_deflated) {
            preconditioner = std::make_unique<schwarz2lvl::DeflatedTwoLevelPreconditioner>(A, prec_1lvl, coarse);
        } else {
            preconditioner = std::make_unique<schwarz2lvl::AdditiveTwoLevelPreconditioner>(prec_1lvl, coarse);
        }

        // 10. Generazione vettori globali e istanziazione del solutore di Krylov astratto
        schwarz2lvl::VectorType x(num_rows);
        schwarz2lvl::VectorType b(num_rows);
        
        x.setZero();
        b.setOnes(); // Vettore di tutti 1 come termine noto fittizio standard

        std::unique_ptr<schwarz2lvl::KrylovSolver> solver;
        if (use_gmres) {
            solver = std::make_unique<schwarz2lvl::KrylovGmres>(400, 1e-6, 30); // Max 400 iterazioni, tolleranza 1e-6, restart 30
        } else {
            solver = std::make_unique<schwarz2lvl::KrylovBicgstab>(400, 1e-6);
        }
        
        // Esecuzione del calcolo con monitoraggio del tempo
        solver->solveWithTiming(A, b, x, *preconditioner);
    }

    // 11. Stampa finale del report dei timer aggregati
    MPI_Barrier(MPI_COMM_WORLD);
    schwarz2lvl::TimerRegistry::instance().report(std::cout, my_rank);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
