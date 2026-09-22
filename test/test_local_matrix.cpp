// Verifica che LocalMatrix::extractLocalSquareBlock() produca ESATTAMENTE
// (stessa sparsity, stessi valori) la A_ii che OneLevelPreconditioner
// (codice esistente, già validato) calcola oggi con la propria logica
// duplicata -- e che i dati per il lumping (somma |valori| esterni sulle
// righe di boundary) coincidano con quanto LocalBlockSplitting calcola.
#include "matrix_market_io.hpp"
#include "global_matrix.hpp"
#include "graph_partitioner.hpp"
#include "subdomain_topology.hpp"
#include "restriction_operator.hpp"
#include "partition_of_unity.hpp"
#include "one_level_preconditioner.hpp"
#include "local_block_splitting.hpp"
#include "local_index_map.hpp"
#include "local_matrix.hpp"
#include <mpi.h>
#include <iostream>
#include <cmath>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int my_rank = 0, num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    std::string matrix_path = argc > 1 ? argv[1] : "matrices/nos5.mtx";

    schwarz2lvl::SparseMatrixWrapper A;
    std::vector<int> partition_map;
    if (my_rank == 0) {
        schwarz2lvl::MatrixMarketIO::readMatrix(matrix_path, A);
        schwarz2lvl::GraphPartitioner partitioner;
        partition_map = partitioner.computePartition(A, num_ranks);
    }
    int num_rows = 0;
    if (my_rank == 0) num_rows = static_cast<int>(A.rows());
    MPI_Bcast(&num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank != 0) { A.getMatrix().resize(num_rows, num_rows); partition_map.resize(num_rows); }
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

    // --- Percorso ESISTENTE, già validato --------------------------------
    schwarz2lvl::SubdomainTopology topology;
    topology.computeTopology(A, partition_map, my_rank);
    schwarz2lvl::RestrictionOperator R_i(topology.getGlobalIndices());
    schwarz2lvl::PartitionOfUnity D_i;
    D_i.computeWeights(A, partition_map, topology);

    schwarz2lvl::OneLevelPreconditioner prec_1lvl;
    prec_1lvl.setup(A, topology, R_i, D_i);
    const schwarz2lvl::MatrixType& A_ii_reference = prec_1lvl.getLocalMatrix();

    // --- Percorso NUOVO ----------------------------------------------------
    schwarz2lvl::LocalIndexMap index_map;
    index_map.build(topology.getGlobalIndices(),
                     static_cast<Eigen::Index>(topology.getInteriorIndices().size()));
    schwarz2lvl::LocalMatrix local_A = schwarz2lvl::buildLocalMatrixFromGlobal(A, index_map);
    schwarz2lvl::MatrixType A_ii_new = local_A.extractLocalSquareBlock();

    // --- Confronto esatto: stessa dimensione, stessa sparsity, stessi valori
    bool same_shape = (A_ii_reference.rows() == A_ii_new.rows()) &&
                       (A_ii_reference.cols() == A_ii_new.cols()) &&
                       (A_ii_reference.nonZeros() == A_ii_new.nonZeros());
    double max_diff = 0.0;
    if (same_shape) {
        schwarz2lvl::MatrixType diff = A_ii_reference - A_ii_new;
        for (int k = 0; k < diff.outerSize(); ++k)
            for (schwarz2lvl::MatrixType::InnerIterator it(diff, k); it; ++it)
                max_diff = std::max(max_diff, std::abs(it.value()));
    }

    // --- Verifica anche i dati per il lumping: per ogni riga di boundary,
    // somma |valori| a colonne NON locali. Confronto tra:
    // (a) LocalBlockSplitting (esistente) applicato su A_tilde_ii = A_ii
    // (b) calcolo diretto dalla riga completa di LocalMatrix::raw()
    schwarz2lvl::MatrixType A_tilde_reference = A_ii_reference;
    schwarz2lvl::LocalBlockSplitting lumper;
    lumper.applyLumping(A, A_tilde_reference, topology);

    const Eigen::Index num_interior = static_cast<Eigen::Index>(topology.getInteriorIndices().size());
    double max_lump_diff = 0.0;
    for (Eigen::Index local_row = num_interior; local_row < local_A.rows(); ++local_row) {
        double external_sum_new = 0.0;
        for (schwarz2lvl::MatrixType::InnerIterator it(local_A.raw(), local_row); it; ++it) {
            if (index_map.globalToLocal(static_cast<int>(it.col())) == -1) {
                external_sum_new += std::abs(it.value());
            }
        }
        double expected_diag = A_ii_reference.coeff(local_row, local_row) - external_sum_new;
        double actual_diag = A_tilde_reference.coeff(local_row, local_row);
        max_lump_diff = std::max(max_lump_diff, std::abs(expected_diag - actual_diag));
    }

    for (int r = 0; r < num_ranks; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r == my_rank) {
            std::cout << "[rank " << my_rank << "] n_i=" << local_A.rows()
                      << " same_shape=" << same_shape
                      << " max_diff_A_ii=" << std::scientific << max_diff
                      << " max_diff_lumping=" << max_lump_diff
                      << ((same_shape && max_diff == 0.0 && max_lump_diff < 1e-14) ? "  >> OK" : "  >> MISMATCH")
                      << std::endl;
            std::cout.flush();
        }
    }

    MPI_Finalize();
    return 0;
}
