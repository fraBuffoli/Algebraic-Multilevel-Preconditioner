/**
 * @file graph_partitioner.hpp
 * @brief Nonoverlapping partition of the adjacency graph G(A + A^T) with METIS.
 */
#ifndef SCHWARZ2LVL_PARTITION_GRAPH_PARTITIONER_HPP
#define SCHWARZ2LVL_PARTITION_GRAPH_PARTITIONER_HPP

#include "csr_matrix.hpp"

#include <string>
#include <vector>

namespace schwarz2lvl::part {

/**
 * @class GraphPartitioner
 * @brief Splits the unknowns into N disjoint subsets Omega_I_i (paper, Section 2).
 *
 * The graph is the *undirected* adjacency graph of A + A^T, as in the paper.
 * With `block_size > 1` the graph is first compressed by cells (a vertex per
 * group of `block_size` consecutive unknowns): every unknown of a cell ends
 * up in the same subdomain, which is what OpenFOAM does for coupled solvers
 * (the mesh, not the matrix, is decomposed).
 */
class GraphPartitioner {
public:
    /// @brief Partitioning options.
    struct Options {
        std::string objective = "cut"; ///< "cut" (edge-cut) or "vol" (communication volume).
        bool vertex_weights = false;   ///< Weight vertices by their number of nonzeros.
        int block_size = 1;            ///< Unknowns per cell.
    };

    /**
     * @brief Computes the partition vector.
     * @param A      Global matrix (rank 0).
     * @param nparts Number of subdomains N.
     * @param opt    Options.
     * @return part[i] in [0, N) for every row i.
     * @throws std::runtime_error if METIS fails.
     */
    static std::vector<int> partition(const CsrMatrix& A, int nparts, const Options& opt);
};

} // namespace schwarz2lvl::part

#endif // SCHWARZ2LVL_PARTITION_GRAPH_PARTITIONER_HPP
