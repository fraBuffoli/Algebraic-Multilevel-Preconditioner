#include "../include/graph_partitioner.hpp"
#include <metis.h>
#include <set>
#include <iostream>
#include <stdexcept>

namespace schwarz2lvl {

std::vector<int> GraphPartitioner::computePartition(const SparseMatrixWrapper& matrix, int num_partitions) const {
    const Eigen::Index n = matrix.rows();
    std::vector<int> partition_map(n, 0);

    // If there is only 1 partition, everyone belongs to subdomain 0 immediately
    if (num_partitions <= 1) {
        return partition_map;
    }

    std::cout << "Building undirected adjacency graph G(A + A^T) for METIS..." << std::endl;

    // To ensure the graph is undirected, we combine the row and column connectivity.
    // std::set automatically eliminates duplicates and keeps indices sorted.
    std::vector<std::set<int>> adjacency_list(n);
    const auto& eigen_mat = matrix.getMatrix();

    for (int k = 0; k < eigen_mat.outerSize(); ++k) {
        for (MatrixType::InnerIterator it(eigen_mat, k); it; ++it) {
            int row = static_cast<int>(it.row());
            int col = static_cast<int>(it.col());

            // Skip diagonal elements as self-loops are not part of the adjacency graph
            if (row != col) {
                adjacency_list[row].insert(col);
                adjacency_list[col].insert(row); // Mirroring for A + A^T
            }
        }
    }

    // Convert the adjacency list into METIS raw CSR format arrays (using METIS native idx_t)
    idx_t nvtxs = static_cast<idx_t>(n);
    idx_t ncon = 1; // Number of balancing constraints
    
    std::vector<idx_t> xadj;
    std::vector<idx_t> adjncy;
    
    xadj.reserve(n + 1);
    xadj.push_back(0);

    for (int i = 0; i < n; ++i) {
        for (int neighbor : adjacency_list[i]) {
            adjncy.push_back(static_cast<idx_t>(neighbor));
        }
        xadj.push_back(static_cast<idx_t>(adjncy.size()));
    }

    // Output array where METIS will write the subdomain mapping
    std::vector<idx_t> part(n, 0);
    idx_t nparts = static_cast<idx_t>(num_partitions);
    idx_t objval = 0;

    // Setup METIS default options
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0; // C-style 0-based indexing

    std::cout << "Invoking METIS_PartGraphKway for " << num_partitions << " partitions..." << std::endl;

    // Call the static library routine compiled via your Makefile
    int status = METIS_PartGraphKway(
        &nvtxs, &ncon, xadj.data(), adjncy.data(),
        nullptr, nullptr, nullptr, &nparts, nullptr,
        nullptr, options, &objval, part.data()
    );

    if (status != METIS_OK) {
        throw std::runtime_error("Error: METIS partitioning failed with status code " + std::to_string(status));
    }

    // Cast the native METIS idx_t array back into our standard std::vector<int>
    for (Eigen::Index i = 0; i < n; ++i) {
        partition_map[i] = static_cast<int>(part[i]);
    }

    std::cout << "METIS successfully cut the graph. Edge-cut value: " << objval << std::endl;
    return partition_map;
}

} // namespace schwarz2lvl
