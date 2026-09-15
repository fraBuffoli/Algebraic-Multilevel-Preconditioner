#include "matrix_market_io.hpp"
#include "sparse_matrix.hpp"
#include <iostream>

int main() {
    const std::string mtx_filepath = "matrices/fidapm05.mtx";

    schwarz2lvl::SparseMatrixWrapper A;

    std::cout << "Loading matrix from resources: " << mtx_filepath << "..." << std::endl;
    bool success = schwarz2lvl::MatrixMarketIO::readMatrix(mtx_filepath, A);

    if (success) {
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "Matrix loaded successfully!" << std::endl;
        std::cout << "Dimensions: " << A.rows() << "x" << A.cols() << std::endl;
        std::cout << "Non-zeros : " << A.nonZeros() << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
    } else {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

