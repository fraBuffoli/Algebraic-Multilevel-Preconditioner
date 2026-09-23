#ifndef DOMAIN_DECOMPOSER_HPP
#define DOMAIN_DECOMPOSER_HPP

#include "config.hpp"
#include "global_matrix.hpp"
#include "subdomain_topology.hpp"
#include "local_matrix.hpp"
#include <string>
#include <vector>

namespace schwarz2lvl {

/**
 * @brief A package of all the data a single rank needs to build its own preconditioner.
 * 
 * Produced entirely on rank 0 and shipped point-to-point to its owner. Contains:
 * - SubdomainTopology topology: the interior/boundary/global index sets for this rank.
 * - LocalMatrix local_matrix: the n_i x n_global representation of the matrix for this rank.
 * - VectorType pou_weights: the partition-of-unity weights for this rank, computed globally by computeGlobalMultiplicities.
 */
struct SubdomainPackage {
    SubdomainTopology topology;
    LocalMatrix local_matrix;
    VectorType pou_weights;  
};

/**
 * @class DomainDecomposer
 * @brief Responsible for reading the global matrix, partitioning it with METIS, and producing SubdomainPackage instances for each rank.
 */
class DomainDecomposer {
public:
    /**
     * @brief Default constructor for DomainDecomposer.
     */
    DomainDecomposer() = default;

    /**
     * @brief Decomposes the global matrix into subdomains and prepares the data for each rank.
     * @param matrix_path The file path to the global matrix in Matrix Market format.
     * @param num_ranks The total number of ranks (subdomains) to decompose the matrix into.
     */
    void decompose(const std::string& matrix_path, int num_ranks);

    Eigen::Index numGlobalRows() const { return n_global_; }
    const std::vector<int>& partitionMap() const { return partition_map_; }
    const SubdomainPackage& packageForRank(int rank) const { return packages_.at(rank); }

private:
    Eigen::Index n_global_ = 0;
    std::vector<int> partition_map_;
    std::vector<SubdomainPackage> packages_;
};

/**
 * @brief Computes the multiplicity of each global node in the matrix.
 * @param A The global sparse matrix.
 * @param partition_map A vector mapping each global node to its owning subdomain.
 * @return A vector where each entry corresponds to the multiplicity of the respective global node.
 */
std::vector<int> computeGlobalMultiplicities(const SparseMatrixWrapper& A,
                                              const std::vector<int>& partition_map);

} // namespace schwarz2lvl

#endif // DOMAIN_DECOMPOSER_HPP