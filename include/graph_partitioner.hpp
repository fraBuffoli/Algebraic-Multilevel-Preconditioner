#ifndef GRAPH_PARTITIONER_HPP
#define GRAPH_PARTITIONER_HPP

#include "sparse_matrix.hpp"
#include <vector>

namespace schwarz2lvl {

/**
 * @class GraphPartitioner
 * @brief Interfaces with the METIS library to compute graph partitioning.
 */
class GraphPartitioner {
public:
    /**
     * @brief Default constructor.
     */
    GraphPartitioner() = default;

    /**
     * @brief Computes a balanced partitioning of the matrix rows using METIS.
     * 
     * This function internally builds the undirected adjacency graph G(A + A^T)
     * by mirroring non-zero patterns, and invokes METIS_PartGraphKway.
     * 
     * @param matrix The global sparse matrix to be partitioned.
     * @param num_partitions The number of desired subdomains (typically equal to MPI size).
     * @return std::vector<int> A vector of size matrix.rows(), where entry [j] is the assigned subdomain ID for row j.
     */
    std::vector<int> computePartition(const SparseMatrixWrapper& matrix, int num_partitions) const;
};

} // namespace schwarz2lvl

#endif // GRAPH_PARTITIONER_HPP
