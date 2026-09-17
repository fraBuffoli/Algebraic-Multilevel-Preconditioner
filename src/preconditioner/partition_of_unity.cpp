#include "partition_of_unity.hpp"
#include <iostream>

namespace schwarz2lvl {

void PartitionOfUnity::computeWeights(const SparseMatrixWrapper& matrix,
                                      const std::vector<int>& partition_map,
                                      const SubdomainTopology& topology) {
    
    const auto& local_indices = topology.getGlobalIndices();
    const Eigen::Index n_i = static_cast<Eigen::Index>(local_indices.size());
    const auto& eigen_mat = matrix.getMatrix();

    // Resize the local weight vector to match the extended subdomain size (n_i)
    weights_.resize(n_i);

    // Loop through all local unknowns (both interior and boundary nodes)
    for (Eigen::Index j = 0; j < n_i; ++j) {
        int global_node = local_indices[j];
        int owner_rank = partition_map[global_node];

        // Case 1: Purely interior nodes. 
        // By definition, METIS assigned them to this rank exclusively.
        // They are not shared in the initial stage, so their weight is exactly 1.0.
        if (j < static_cast<Eigen::Index>(topology.getInteriorIndices().size())) {
            weights_(j) = 1.0;
        } 
        // Case 2: Overlapping boundary nodes.
        // We need to check how many different subdomains actually connect to this node.
        else {
            // Count how many subdomains share this boundary node.
            // It is shared by its original owner plus any subdomain that touches it.
            std::vector<int> sharing_subdomains;
            sharing_subdomains.push_back(owner_rank);

            // Scan the column connections of this specific row in the global matrix
            for (MatrixType::InnerIterator it(eigen_mat, global_node); it; ++it) {
                int neighbor_col = static_cast<int>(it.col());
                int neighbor_rank = partition_map[neighbor_col];

                // If a neighbor belongs to a new rank, this rank shares the node
                if (std::find(sharing_subdomains.begin(), sharing_subdomains.end(), neighbor_rank) == sharing_subdomains.end()) {
                    sharing_subdomains.push_back(neighbor_rank);
                }
            }

            // The partition of unity weight is the reciprocal of the number of sharing subdomains
            double num_sharers = static_cast<double>(sharing_subdomains.size());
            weights_(j) = 1.0 / num_sharers;
        }
    }
}

} // namespace schwarz2lvl
