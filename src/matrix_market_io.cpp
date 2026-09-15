#include "../include/matrix_market_io.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>

namespace schwarz2lvl {

bool MatrixMarketIO::readMatrix(const std::string& filename, SparseMatrixWrapper& out_matrix) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open Matrix Market file: " << filename << std::endl;
        return false;
    }

    std::string header;
    if (!std::getline(file, header)) {
        std::cerr << "Error: Empty file." << std::endl;
        return false;
    }

    // Parse matrix market header banners
    if (header.find("%%MatrixMarket") == std::string::npos) {
        std::cerr << "Error: Invalid Matrix Market banner." << std::endl;
        return false;
    }

    bool is_symmetric = (header.find("symmetric") != std::string::npos);
    bool is_coordinate = (header.find("coordinate") != std::string::npos);
    bool is_real = (header.find("real") != std::string::npos) || (header.find("integer") != std::string::npos);

    if (!is_coordinate || !is_real) {
        std::cerr << "Error: Only real/integer coordinate matrices are supported by this parser." << std::endl;
        return false;
    }

    // Skip comments until the size line
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') {
            continue;
        }
        break; // Found the size line
    }

    std::stringstream ss(line);
    Eigen::Index num_rows = 0, num_cols = 0, num_nonzeros = 0;
    if (!(ss >> num_rows >> num_cols >> num_nonzeros)) {
        std::cerr << "Error: Invalid Matrix Market size specification." << std::endl;
        return false;
    }

    // Pre-allocate triplet list for optimal construction performance
    std::vector<Eigen::Triplet<double>> triplet_list;
    triplet_list.reserve(is_symmetric ? num_nonzeros * 2 : num_nonzeros);

    Eigen::Index r, c;
    double val;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') continue;
        
        std::stringstream line_ss(line);
        if (line_ss >> r >> c >> val) {
            // Matrix Market uses 1-based indexing; convert to C++ 0-based indexing
            triplet_list.push_back(Eigen::Triplet<double>(r - 1, c - 1, val));
            
            // If the matrix banner specifies symmetric, explicitly mirror the entry
            if (is_symmetric && (r != c)) {
                triplet_list.push_back(Eigen::Triplet<double>(c - 1, r - 1, val));
            }
        }
    }

    file.close();

    // Populate and compress the Eigen structures
    out_matrix.getMatrix().resize(num_rows, num_cols);
    out_matrix.getMatrix().setFromTriplets(triplet_list.begin(), triplet_list.end());
    out_matrix.makeCompressed();

    std::cout << "Successfully loaded matrix: " << num_rows << "x" << num_cols 
              << " with " << out_matrix.nonZeros() << " non-zero elements." << std::endl;

    return true;
}

} // namespace schwarz2lvl
