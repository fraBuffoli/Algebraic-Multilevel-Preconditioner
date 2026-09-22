// Standalone validation for HaloExchange: builds the SAME decomposition as
// main.cpp (matrix, METIS partition, topology per rank), but keeps the full
// global vector around as ground truth -- something we can still do today,
// and will NOT be able to do anymore once the matrix stops being replicated
// (Step 2+). This is exactly the moment to cross-check HaloExchange against
// a known-correct reference before we start depending on it.
#include "matrix_market_io.hpp"
#include "global_matrix.hpp"
#include "graph_partitioner.hpp"
#include "subdomain_topology.hpp"
#include "restriction_operator.hpp"
#include "halo_exchange.hpp"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>

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

    schwarz2lvl::SubdomainTopology topology;
    topology.computeTopology(A, partition_map, my_rank);
    schwarz2lvl::RestrictionOperator R_i(topology.getGlobalIndices());

    schwarz2lvl::HaloExchange halo;
    halo.setup(partition_map, topology);

    // Ground truth: a full global vector with a DISTINCT, checkable value
    // per entry (global_id * 1000 + 7), still built the "old" way since A is
    // still fully replicated at this point.
    schwarz2lvl::VectorType global_ref(num_rows);
    for (int i = 0; i < num_rows; ++i) global_ref(i) = static_cast<double>(i) * 1000.0 + 7.0;

    // Reference local vector: direct restriction from the (still available)
    // global vector -- this is exactly what RestrictionOperator::apply does
    // today, and is by definition correct.
    schwarz2lvl::VectorType local_reference;
    R_i.apply(global_ref, local_reference);

    // Test vector: fill ONLY the interior part correctly (this rank's own
    // authoritative data), corrupt the boundary part on purpose (-999), then
    // ask HaloExchange to fix it using ONLY point-to-point neighbor comms.
    const Eigen::Index n_i = R_i.localSize();
    const Eigen::Index num_interior = static_cast<Eigen::Index>(topology.getInteriorIndices().size());
    schwarz2lvl::VectorType local_test = schwarz2lvl::VectorType::Constant(n_i, -999.0);
    for (Eigen::Index j = 0; j < num_interior; ++j) local_test(j) = local_reference(j);

    halo.exchange(local_test);

    const double max_err = (local_test - local_reference).cwiseAbs().maxCoeff();

    // Also check nothing pathological: interior part must be untouched.
    const double interior_err = (local_test.head(num_interior) - local_reference.head(num_interior)).cwiseAbs().maxCoeff();

    for (int r = 0; r < num_ranks; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r == my_rank) {
            std::cout << "[rank " << my_rank << "] n_i=" << n_i
                      << " num_interior=" << num_interior
                      << " neighbors=" << halo.numNeighbors()
                      << " boundary_max_err=" << std::scientific << max_err
                      << " interior_max_err=" << interior_err
                      << (max_err == 0.0 && interior_err == 0.0 ? "  >> OK" : "  >> MISMATCH")
                      << std::endl;
            std::cout.flush();
        }
    }

    MPI_Finalize();
    return 0;
}