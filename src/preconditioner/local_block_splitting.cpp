#include "local_block_splitting.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace schwarz2lvl {

void LocalBlockSplitting::applyLumping(const SparseMatrixWrapper& global_A,
                                       MatrixType& local_A_ii,
                                       const SubdomainTopology& topology) const {
    
    const auto& g_mat = global_A.getMatrix();
    const auto& local_indices = topology.getGlobalIndices();
    
    const Eigen::Index num_interiors = static_cast<Eigen::Index>(topology.getInteriorIndices().size());
    const Eigen::Index n_i = local_A_ii.rows();

    // 1. Build a quick look-up table to check if a global column belongs to our extended subdomain.
    std::vector<bool> is_local_column(global_A.rows(), false);
    for (Eigen::Index j = 0; j < n_i; ++j) {
        is_local_column[local_indices[j]] = true;
    }

    // 2. Loop exclusively over the boundary/overlap rows.
    // Thanks to [Interiors, Boundary] ordering, boundary rows start exactly at index 'num_interiors'.
    for (Eigen::Index local_row = num_interiors; local_row < n_i; ++local_row) {
        int global_row = local_indices[local_row];
        double external_sum = 0.0;

        // Scan the entire global row pattern to find connections pointing OUTSIDE our extended subdomain
        for (MatrixType::InnerIterator it(g_mat, global_row); it; ++it) {
            int global_col = static_cast<int>(it.col());

            // If the column does NOT belong to our extended subdomain, it's an external connection
            if (!is_local_column[global_col]) {
                external_sum += std::abs(it.value());
            }
        }

        // 3. Modify the diagonal entry of the local matrix A_ii in-place.
        // We look for the diagonal element (where local_col == local_row) inside the local row.
        for (MatrixType::InnerIterator it(local_A_ii, local_row); it; ++it) {
            if (it.col() == local_row) {
                // Subtract the external connections sum from the local diagonal coefficient
                it.valueRef() -= external_sum;
                break; 
            }
        }
    }
}

void LocalBlockSplitting::applyLumping(const LocalMatrix& local_A,
                                       MatrixType& local_A_ii) const {
    const MatrixType& full_rows = local_A.raw();
    const LocalIndexMap& index_map = local_A.indexMap();
    const Eigen::Index n_i = index_map.size();

    if (local_A_ii.rows() != n_i || local_A_ii.cols() != n_i) {
        throw std::invalid_argument("LocalBlockSplitting::applyLumping: local_A_ii is not n_i x n_i ");
    }

    for (Eigen::Index local_row = index_map.numInterior(); local_row < n_i; ++local_row) {
        double external_sum = 0.0;
        for (MatrixType::InnerIterator it(full_rows, local_row); it; ++it) {
            if (index_map.globalToLocal(static_cast<int>(it.col())) == -1) {
                external_sum += std::abs(it.value());
            }
        }
        for (MatrixType::InnerIterator it(local_A_ii, local_row); it; ++it) {
            if (it.col() == local_row) { it.valueRef() -= external_sum; break; }
        }
    }
}

} // namespace schwarz2lvl
