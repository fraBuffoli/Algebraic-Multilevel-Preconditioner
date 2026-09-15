/**
 * @file sparse_matrix.hpp
 * @brief Definition of sparse matrix and dense vector structures using Eigen for real numbers.
 */

#ifndef SPARSE_MATRIX_HPP
#define SPARSE_MATRIX_HPP

#include "config.hpp"

namespace schwarz2lvl {

/**
 * @class SparseMatrixWrapper
 * @brief Manages a global or local sparse matrix in Compressed Sparse Row (CSR) format using double precision.
 */
class SparseMatrixWrapper {
public:

    /**
     * @brief Default constructor creating an empty sparse matrix.
     */
    SparseMatrixWrapper() = default;

    /**
     * @brief Constructs a sparse matrix with a predefined size.
     * @param rows Number of rows.
     * @param cols Number of columns.
     */
    SparseMatrixWrapper(Eigen::Index rows, Eigen::Index cols) : matrix_(rows, cols) {}

    /**
     * @brief Non-const access to the underlying Eigen sparse matrix.
     * @return Reference to the Eigen::SparseMatrix.
     */
    MatrixType& getMatrix() { return matrix_; }

    /**
     * @brief Const access to the underlying Eigen sparse matrix.
     * @return Const reference to the Eigen::SparseMatrix.
     */
    const MatrixType& getMatrix() const { return matrix_; }

    /**
     * @brief Gets the number of rows in the matrix.
     * @return The row dimension.
     */
    Eigen::Index rows() const { return matrix_.rows(); }

    /**
     * @brief Gets the number of columns in the matrix.
     * @return The column dimension.
     */
    Eigen::Index cols() const { return matrix_.cols(); }

    /**
     * @brief Gets the number of non-zero elements stored in the matrix.
     * @return Number of non-zeros.
     */
    Eigen::Index nonZeros() const { return matrix_.nonZeros(); }

    /**
     * @brief Compresses the underlying storage to optimize memory layout and performance.
     */
    void makeCompressed() { matrix_.makeCompressed(); }

private:
    MatrixType matrix_; // The actual Eigen sparse matrix object.
};

} // namespace schwarz2lvl

#endif // SPARSE_MATRIX_HPP
