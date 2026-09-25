/**
 * @file csr_matrix.hpp
 * @brief Plain CSR container for the *global* matrix, used on rank 0 only
 *        during the decomposition phase.
 */
#ifndef SCHWARZ2LVL_PARTITION_CSR_MATRIX_HPP
#define SCHWARZ2LVL_PARTITION_CSR_MATRIX_HPP

#include "types.hpp"

#include <cstdint>
#include <vector>

namespace schwarz2lvl::part {

/**
 * @brief Compressed Sparse Row matrix with 64-bit indices.
 *
 * Columns inside each row are sorted and unique. This container is used
 * only by the rank-0 decomposition (reading, partitioning, permuting and
 * scattering the matrix); it is never used by the preconditioner.
 */
struct CsrMatrix {
    core::GlobalIndex nrows = 0;                ///< Number of rows.
    core::GlobalIndex ncols = 0;                ///< Number of columns.
    std::vector<core::GlobalIndex> row_ptr;     ///< Row pointer (size nrows+1).
    std::vector<core::GlobalIndex> col;         ///< Column indices.
    std::vector<core::Scalar> val;              ///< Values.
    bool symmetric = false;                 ///< True if the matrix is known to be symmetric.

    /// @brief Number of stored nonzeros.
    core::GlobalIndex nnz() const { return row_ptr.empty() ? 0 : row_ptr.back(); }

    /**
     * @brief Builds the CSR matrix from coordinate entries (duplicates are summed).
     * @param nrows Number of rows.
     * @param ncols Number of columns.
     * @param rows  Row indices (0-based).
     * @param cols  Column indices (0-based).
     * @param vals  Values.
     * @return The assembled matrix with sorted, unique columns.
     */
    static CsrMatrix fromCoo(core::GlobalIndex nrows, core::GlobalIndex ncols, const std::vector<core::GlobalIndex>& rows,
                             const std::vector<core::GlobalIndex>& cols, const std::vector<core::Scalar>& vals);

    /**
     * @brief Sparse matrix-vector product y = A x (sequential).
     * @param x Input vector (size ncols).
     * @param y Output vector (size nrows).
     */
    void multiply(const std::vector<core::Scalar>& x, std::vector<core::Scalar>& y) const;
};

} // namespace schwarz2lvl::part

#endif // SCHWARZ2LVL_PARTITION_CSR_MATRIX_HPP
