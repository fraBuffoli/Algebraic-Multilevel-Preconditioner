#include "subdomain_topology.hpp"
#include <algorithm>
#include <iostream>

namespace schwarz2lvl {

void SubdomainTopology::computeTopology(const SparseMatrixWrapper& matrix, 
                                         const std::vector<int>& partition_map, 
                                         int target_rank) {
    const Eigen::Index n = matrix.rows();
    const auto& eigen_mat = matrix.getMatrix();

    // Reset local containers to ensure clean state
    interior_indices_.clear();
    boundary_indices_.clear();
    global_indices_.clear();

    // 1. Extract purely interior nodes assigned to this rank by METIS
    for (Eigen::Index i = 0; i < n; ++i) {
        if (partition_map[i] == target_rank) {
            interior_indices_.push_back(static_cast<int>(i));
        }
    }

    // 2. Find overlap nodes at distance 1 using a boolean marker array.
    // This tracks whether a global node has already been added to the boundary
    // to guarantee O(1) duplicate checks and preserve linear time complexity.
    std::vector<bool> is_already_boundary(n, false);

    // Scan the sparsity pattern of the rows owned by this rank
    for (int int_node : interior_indices_) {
        // InnerIterator directly reads non-zero entries of row 'int_node' in CSR layout
        for (MatrixType::InnerIterator it(eigen_mat, int_node); it; ++it) {
            int neighbor_col = static_cast<int>(it.col());

            // We look for connections where the column belongs to ANOTHER rank
            if (partition_map[neighbor_col] != target_rank) {
                // If we haven't processed this external node yet, add it to the boundary
                if (!is_already_boundary[neighbor_col]) {
                    is_already_boundary[neighbor_col] = true;
                    boundary_indices_.push_back(neighbor_col);
                }
            }
        }
    }

    // 3. Ensure the boundary indices are sorted to guarantee algebraic consistency
    std::sort(boundary_indices_.begin(), boundary_indices_.end());

    // 4. Assemble the final complete list of local unknowns: [Interior, Boundary]
    global_indices_.reserve(interior_indices_.size() + boundary_indices_.size());
    global_indices_.insert(global_indices_.end(), interior_indices_.begin(), interior_indices_.end());
    global_indices_.insert(global_indices_.end(), boundary_indices_.begin(), boundary_indices_.end());
}

void SubdomainTopology::setIndices(std::vector<int> interior_indices, std::vector<int> boundary_indices) {
    interior_indices_ = std::move(interior_indices);
    boundary_indices_ = std::move(boundary_indices);
    global_indices_.clear();
    global_indices_.reserve(interior_indices_.size() + boundary_indices_.size());
    global_indices_.insert(global_indices_.end(), interior_indices_.begin(), interior_indices_.end());
    global_indices_.insert(global_indices_.end(), boundary_indices_.begin(), boundary_indices_.end());
}

} // namespace schwarz2lvl
