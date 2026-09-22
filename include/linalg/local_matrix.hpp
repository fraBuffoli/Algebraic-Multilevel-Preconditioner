#ifndef LOCAL_MATRIX_HPP
#define LOCAL_MATRIX_HPP

#include "config.hpp"
#include "global_matrix.hpp"
#include "local_index_map.hpp"
#include <vector>

namespace schwarz2lvl {

/**
 * @class LocalMatrix
 * @brief Represents the local matrix for a subdomain in a parallel computation.
 */
class LocalMatrix {
public:
    /**
     * @brief Default constructor for LocalMatrix.
     * Initializes an empty LocalMatrix.
     */
    LocalMatrix() = default;

    /**
     * @brief Build the local matrix from a local index map and a list of triplets.
     * @param index_map The LocalIndexMap that defines the mapping between local and global indices.
     * @param n_global The total number of global columns in the matrix.
     * @param local_triplets A vector of Eigen::Triplet<double> representing the non-zero entries of the local matrix. Each triplet contains a local row index, a global column index, and a value.
     */
    void build(const LocalIndexMap& index_map,
               Eigen::Index n_global,
               const std::vector<Eigen::Triplet<double>>& local_triplets);

    Eigen::Index rows() const { return matrix_.rows(); }      // = n_i
    Eigen::Index globalCols() const { return n_global_; }      // = n_global

    const MatrixType& raw() const { return matrix_; }          // n_i x n_global
    const LocalIndexMap& indexMap() const { return index_map_; }
    
    /**
     * @brief Extract the local square block of the matrix corresponding to interior indices.
     * @return A MatrixType representing the local square block of size n_i x n_i, containing only the columns that are also rows in the local subdomain 
     */
    MatrixType extractLocalSquareBlock() const;

private:
    MatrixType matrix_;
    LocalIndexMap index_map_;
    Eigen::Index n_global_ = 0;
};

/**
 * @brief Build a LocalMatrix from a global matrix and a local index map.
 * @param global_A The SparseMatrixWrapper representing the global matrix.
 * @param index_map The LocalIndexMap that defines the mapping between local and global indices.
 * @return A LocalMatrix containing the rows of the global matrix corresponding to the local indices defined in index_map, with all non-zero columns included.
 */
LocalMatrix buildLocalMatrixFromGlobal(const SparseMatrixWrapper& global_A,
                                        const LocalIndexMap& index_map);

} // namespace schwarz2lvl

#endif // LOCAL_MATRIX_HPP
