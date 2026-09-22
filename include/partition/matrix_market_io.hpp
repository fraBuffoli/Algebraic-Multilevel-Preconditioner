#ifndef MATRIX_MARKET_IO_HPP
#define MATRIX_MARKET_IO_HPP

#include "global_matrix.hpp"
#include <string>

namespace schwarz2lvl {

/**
 * @class MatrixMarketIO
 * @brief Utilities for reading Matrix Market files (.mtx) into SparseMatrixWrapper.
 */    
class MatrixMarketIO {
public:
    /**
     * @brief Reads a Matrix Market file and populates a SparseMatrixWrapper.
     * 
     * Supports coordinate format with real numbers. Handles both 'general' 
     * and 'symmetric' storage types.
     * 
     * @param filename Path to the .mtx file.
     * @param out_matrix Reference to the SparseMatrixWrapper object to be filled.
     * @return true if the file was read successfully, false otherwise.
     */
    static bool readMatrix(const std::string& filename, SparseMatrixWrapper& out_matrix);
};

} // namespace schwarz2lvl

#endif // MATRIX_MARKET_IO_HPP
