#include "domain_decomposer.hpp"
#include "matrix_market_io.hpp"
#include "graph_partitioner.hpp"
#include "local_index_map.hpp"
#include <mpi.h>
#include <iostream>
#include <stdexcept>
#include <set>

namespace schwarz2lvl {

std::vector<int> computeGlobalMultiplicities(const SparseMatrixWrapper& A,
                                              const std::vector<int>& partition_map) {
    const Eigen::Index n = A.rows();
    const auto& mat = A.getMatrix();

    std::vector<std::set<int>> reached_by(n);
    for (Eigen::Index u = 0; u < n; ++u) {
        const int owner = partition_map[u];
        for (MatrixType::InnerIterator it(mat, u); it; ++it) {
            const int v = static_cast<int>(it.col());
            if (partition_map[v] != owner) reached_by[v].insert(owner);
        }
    }

    std::vector<int> multiplicity(n, 1);
    for (Eigen::Index v = 0; v < n; ++v) multiplicity[v] = 1 + static_cast<int>(reached_by[v].size());
    return multiplicity;
}

void DomainDecomposer::decompose(const std::string& matrix_path, int num_ranks) {
    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (my_rank != 0) {
        std::cerr << "DomainDecomposer::decompose Error: called on rank " << my_rank
                  << " -- this class must run on rank 0 only." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    SparseMatrixWrapper A;
    if (!MatrixMarketIO::readMatrix(matrix_path, A)) {
        throw std::runtime_error("DomainDecomposer Error: failed to read matrix from " + matrix_path);
    }
    n_global_ = A.rows();

    GraphPartitioner partitioner;
    partition_map_ = partitioner.computePartition(A, num_ranks);

    std::cout << "DomainDecomposer: computing global node multiplicities (single pass)..." << std::endl;
    const std::vector<int> multiplicity = computeGlobalMultiplicities(A, partition_map_);

    packages_.clear();
    packages_.resize(num_ranks);

    for (int r = 0; r < num_ranks; ++r) {
        SubdomainPackage& pkg = packages_[r];

        pkg.topology.computeTopology(A, partition_map_, r);

        LocalIndexMap index_map;
        index_map.build(pkg.topology.getGlobalIndices(),
                         static_cast<Eigen::Index>(pkg.topology.getInteriorIndices().size()));

        pkg.local_matrix = buildLocalMatrixFromGlobal(A, index_map);

        const auto& global_indices = pkg.topology.getGlobalIndices();
        pkg.pou_weights.resize(static_cast<Eigen::Index>(global_indices.size()));
        for (size_t j = 0; j < global_indices.size(); ++j) {
            pkg.pou_weights(static_cast<Eigen::Index>(j)) =
                1.0 / static_cast<double>(multiplicity[global_indices[j]]);
        }

        std::cout << "DomainDecomposer: rank " << r << " package ready, n_i=" << pkg.local_matrix.rows()
                  << " (interior=" << pkg.topology.getInteriorIndices().size()
                  << ", boundary=" << pkg.topology.getBoundaryIndices().size() << ")" << std::endl;
    }
}

} // namespace schwarz2lvl