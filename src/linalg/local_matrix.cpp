#include "local_matrix.hpp"

namespace schwarz2lvl {

void LocalMatrix::build(const LocalIndexMap& index_map,
                        Eigen::Index n_global,
                        const std::vector<Eigen::Triplet<double>>& local_triplets) {
    index_map_ = index_map;
    n_global_ = n_global;
    matrix_.resize(index_map_.size(), n_global_);
    matrix_.setFromTriplets(local_triplets.begin(), local_triplets.end());
    matrix_.makeCompressed();
}

MatrixType LocalMatrix::extractLocalSquareBlock() const {
    const Eigen::Index n_i = rows();
    std::vector<Eigen::Triplet<double>> square_triplets;
    square_triplets.reserve(static_cast<size_t>(matrix_.nonZeros()));

    for (Eigen::Index local_row = 0; local_row < n_i; ++local_row) {
        for (MatrixType::InnerIterator it(matrix_, local_row); it; ++it) {
            const int global_col = static_cast<int>(it.col());
            const int local_col = index_map_.globalToLocal(global_col);
            if (local_col != -1) {
                square_triplets.emplace_back(local_row, local_col, it.value());
            }
        }
    }

    MatrixType A_ii(n_i, n_i);
    A_ii.setFromTriplets(square_triplets.begin(), square_triplets.end());
    A_ii.makeCompressed();
    return A_ii;
}

LocalMatrix buildLocalMatrixFromGlobal(const SparseMatrixWrapper& global_A,
                                        const LocalIndexMap& index_map) {
    const auto& global_mat = global_A.getMatrix();
    const Eigen::Index n_i = index_map.size();

    std::vector<Eigen::Triplet<double>> triplets;
    Eigen::Index nnz_reserve = 0;
    for (Eigen::Index local_row = 0; local_row < n_i; ++local_row) {
        const int global_row = index_map.localToGlobal(local_row);
        nnz_reserve += global_mat.innerVector(global_row).nonZeros();
    }
    triplets.reserve(static_cast<size_t>(nnz_reserve));

    for (Eigen::Index local_row = 0; local_row < n_i; ++local_row) {
        const int global_row = index_map.localToGlobal(local_row);
        for (MatrixType::InnerIterator it(global_mat, global_row); it; ++it) {
            triplets.emplace_back(local_row, static_cast<int>(it.col()), it.value());
        }
    }

    LocalMatrix local_A;
    local_A.build(index_map, global_A.rows(), triplets);
    return local_A;
}

} // namespace schwarz2lvl
