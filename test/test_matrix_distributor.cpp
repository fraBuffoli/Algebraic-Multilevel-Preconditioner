// Verifica MatrixDistributor: SOLO rank 0 chiama DomainDecomposer::decompose,
// ogni rank riceve il proprio pacchetto via MatrixDistributor::distribute, e
// lo confronta contro un riferimento indipendente -- ogni rank rilegge il
// file .mtx per conto proprio (I/O ridondante accettabile SOLO qui, per
// validazione: e' esattamente cio' che DomainDecomposer esiste per evitare
// in produzione) e ricalcola topologia/A_ii/pesi POU con la pipeline
// esistente e gia' validata, usando la STESSA partition_map che
// DomainDecomposer ha calcolato (ricevuta via broadcast, economico: O(n)
// interi, non la matrice).
#include "matrix_market_io.hpp"
#include "global_matrix.hpp"
#include "subdomain_topology.hpp"
#include "restriction_operator.hpp"
#include "partition_of_unity.hpp"
#include "one_level_preconditioner.hpp"
#include "domain_decomposer.hpp"
#include "matrix_distributor.hpp"
#include <mpi.h>
#include <iostream>
#include <cmath>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int my_rank = 0, num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    const std::string matrix_path = argc > 1 ? argv[1] : "matrices/nos5.mtx";

    schwarz2lvl::DomainDecomposer decomposer;
    if (my_rank == 0) decomposer.decompose(matrix_path, num_ranks);

    schwarz2lvl::SubdomainPackage my_pkg =
        schwarz2lvl::MatrixDistributor::distribute(my_rank == 0 ? &decomposer : nullptr);

    std::vector<int> partition_map;
    int num_rows = 0;
    if (my_rank == 0) {
        partition_map = decomposer.partitionMap();
        num_rows = static_cast<int>(decomposer.numGlobalRows());
    }
    MPI_Bcast(&num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank != 0) partition_map.resize(num_rows);
    MPI_Bcast(partition_map.data(), num_rows, MPI_INT, 0, MPI_COMM_WORLD);

    schwarz2lvl::SparseMatrixWrapper A_ref;
    schwarz2lvl::MatrixMarketIO::readMatrix(matrix_path, A_ref);

    schwarz2lvl::SubdomainTopology topo_ref;
    topo_ref.computeTopology(A_ref, partition_map, my_rank);
    schwarz2lvl::RestrictionOperator R_ref(topo_ref.getGlobalIndices());
    schwarz2lvl::PartitionOfUnity D_ref;
    D_ref.computeWeights(A_ref, partition_map, topo_ref);
    schwarz2lvl::OneLevelPreconditioner prec_ref;
    prec_ref.setup(A_ref, topo_ref, R_ref, D_ref);
    const schwarz2lvl::MatrixType& A_ii_ref = prec_ref.getLocalMatrix();

    schwarz2lvl::MatrixType A_ii_new = my_pkg.local_matrix.extractLocalSquareBlock();

    bool same_shape = (A_ii_ref.rows() == A_ii_new.rows()) &&
                       (A_ii_ref.cols() == A_ii_new.cols()) &&
                       (A_ii_ref.nonZeros() == A_ii_new.nonZeros());
    double max_diff_Aii = 0.0;
    if (same_shape) {
        schwarz2lvl::MatrixType diff = A_ii_ref - A_ii_new;
        for (int k = 0; k < diff.outerSize(); ++k)
            for (schwarz2lvl::MatrixType::InnerIterator it(diff, k); it; ++it)
                max_diff_Aii = std::max(max_diff_Aii, std::abs(it.value()));
    }

    bool same_topology = (topo_ref.getGlobalIndices() == my_pkg.topology.getGlobalIndices()) &&
                          (topo_ref.getInteriorIndices() == my_pkg.topology.getInteriorIndices()) &&
                          (topo_ref.getBoundaryIndices() == my_pkg.topology.getBoundaryIndices());

    bool same_size_pou = (D_ref.getWeights().size() == my_pkg.pou_weights.size());
    double max_diff_pou = same_size_pou ? (D_ref.getWeights() - my_pkg.pou_weights).cwiseAbs().maxCoeff() : -1.0;

    bool ok = same_shape && max_diff_Aii == 0.0 && same_topology && same_size_pou && max_diff_pou < 1e-14;

    for (int r = 0; r < num_ranks; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r == my_rank) {
            std::cout << "[rank " << my_rank << "] n_i=" << my_pkg.local_matrix.rows()
                      << " same_shape=" << same_shape
                      << " max_diff_Aii=" << std::scientific << max_diff_Aii
                      << " same_topology=" << same_topology
                      << " max_diff_pou=" << max_diff_pou
                      << (ok ? "  >> OK" : "  >> MISMATCH") << std::endl;
            std::cout.flush();
        }
    }

    MPI_Finalize();
    return ok ? 0 : 1;
}