/**
 * @file matrix_market_io.hpp
 * @brief Sequential Matrix Market reader/writer (used on rank 0).
 */
#ifndef SCHWARZ2LVL_PARTITION_MATRIX_MARKET_IO_HPP
#define SCHWARZ2LVL_PARTITION_MATRIX_MARKET_IO_HPP

#include "csr_matrix.hpp"

#include <string>
#include <vector>

namespace schwarz2lvl::part {

/**
 * @class MatrixMarketIO
 * @brief Matrix Market input/output.
 *
 * Supported: `matrix coordinate real|integer|pattern general|symmetric|skew-symmetric`
 * and `matrix array real general` (dense vectors). Complex matrices are rejected
 * (this implementation works in real arithmetic only).
 */
class MatrixMarketIO {
public:
    /**
     * @brief Reads a sparse matrix. Symmetric storage is expanded to full storage.
     * @param path File name.
     * @return The matrix in CSR format (CsrMatrix::symmetric set from the header).
     * @throws std::runtime_error on I/O or format errors.
     */
    static CsrMatrix readMatrix(const std::string& path);

    /**
     * @brief Reads a dense vector (array format, or coordinate format with one column).
     * @param path File name.
     * @return The vector.
     */
    static std::vector<core::Scalar> readVector(const std::string& path);

    /**
     * @brief Writes a sparse matrix in coordinate general format.
     * @param path File name.
     * @param A    Matrix.
     */
    static void writeMatrix(const std::string& path, const CsrMatrix& A);

    /**
     * @brief Writes a dense vector in array format.
     * @param path File name.
     * @param v    Vector.
     */
    static void writeVector(const std::string& path, const std::vector<core::Scalar>& v);
};

} // namespace schwarz2lvl::part

#endif // SCHWARZ2LVL_PARTITION_MATRIX_MARKET_IO_HPP
