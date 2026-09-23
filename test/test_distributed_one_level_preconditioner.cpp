// Verifica DistributedOneLevelPreconditioner confrontando z_owned (vettori
// OWNED, halo mirato) contro OneLevelPreconditioner::apply (vettori globali
// replicati, MPI_Allreduce su n) -- stesso identico calcolo, due percorsi
// di comunicazione diversi. Ogni rank rilegge il file per conto proprio
// (I/O ridondante accettabile SOLO qui, per validazione) e usa la STESSA
// partition_map di DomainDecomposer (ricevuta via broadcast, O(n) interi)
// per costruire il riferimento -- necessario perche'
// OneLevelPreconditioner::apply e' collettivo su MPI_Allreduce(n), quindi
// va chiamato allo stesso modo su ogni rank.
#include "matrix_market_io.hpp"
#include "global_matrix.hpp"
#include "subdomain_topology.hpp"
#include "restriction_operator.hpp"
#include "partition_of_unity.hpp"
#include "one_level_preconditioner.hpp"
#include "domain_decomposer.hpp"
#include "matrix_distributor.hpp"
#include "halo_exchange.hpp"
#include "distributed_one_level_preconditioner.hpp"
#include <mpi.h>
#include <iostream>
#include <cmath>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int my_rank = 0, num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    const std::string matrix_path = argc > 1 ? argv[1] : "matrices/nos5.mtx";

    // --- Percorso NUOVO: decompose + distribute (gia' validati) --------
    schwarz2lvl::DomainDecomposer decomposer;
    if (my_rank == 0) decomposer.decompose(matrix_path, num_ranks);
    schwarz2lvl::SubdomainPackage pkg =
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

    schwarz2lvl::HaloExchange halo;
    halo.setup(partition_map, pkg.topology);

    schwarz2lvl::DistributedOneLevelPreconditioner prec_new;
    prec_new.setup(pkg.local_matrix, halo, pkg.pou_weights);

    // --- Vettore di test globale (valori distinti, non banali) --------
    schwarz2lvl::VectorType r_global(num_rows);
    for (int i = 0; i < num_rows; ++i) r_global(i) = std::sin(0.013 * i) + 0.002 * i;

    const Eigen::Index num_interior = halo.numInterior();
    schwarz2lvl::VectorType r_owned(num_interior);
    for (Eigen::Index j = 0; j < num_interior; ++j) {
        r_owned(j) = r_global(pkg.topology.getInteriorIndices()[j]);
    }

    schwarz2lvl::VectorType z_owned = schwarz2lvl::VectorType::Zero(num_interior);
    prec_new.apply(r_owned, z_owned);

    // --- Riferimento: pipeline esistente, collettiva su tutti i rank ---
    schwarz2lvl::SparseMatrixWrapper A_ref;
    schwarz2lvl::MatrixMarketIO::readMatrix(matrix_path, A_ref);

    schwarz2lvl::SubdomainTopology topo_ref;
    topo_ref.computeTopology(A_ref, partition_map, my_rank);
    schwarz2lvl::RestrictionOperator R_ref(topo_ref.getGlobalIndices());
    schwarz2lvl::PartitionOfUnity D_ref;
    D_ref.computeWeights(A_ref, partition_map, topo_ref);
    schwarz2lvl::OneLevelPreconditioner prec_ref;
    prec_ref.setup(A_ref, topo_ref, R_ref, D_ref);

    schwarz2lvl::VectorType z_global_ref = schwarz2lvl::VectorType::Zero(num_rows);
    prec_ref.apply(r_global, z_global_ref);  // collettivo: MPI_Allreduce(n) dentro

    double max_err = 0.0, max_abs = 0.0;
    const auto& my_interior = pkg.topology.getInteriorIndices();
    for (Eigen::Index j = 0; j < num_interior; ++j) {
        const double ref = z_global_ref(my_interior[j]);
        max_err = std::max(max_err, std::abs(z_owned(j) - ref));
        max_abs = std::max(max_abs, std::abs(ref));
    }
    const bool ok = max_err < 1e-10;

    for (int r = 0; r < num_ranks; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r == my_rank) {
            std::cout << "[rank " << my_rank << "] num_interior=" << num_interior
                      << " max_err_abs=" << std::scientific << max_err
                      << " (max|z_ref|=" << max_abs << ")"
                      << (ok ? "  >> OK" : "  >> MISMATCH") << std::endl;
            std::cout.flush();
        }
    }

    MPI_Finalize();
    return ok ? 0 : 1;
}