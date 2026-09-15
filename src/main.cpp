#include "matrix_market_io.hpp"
#include "sparse_matrix.hpp"
#include "graph_partitioner.hpp"
#include <iostream>

int main() {
    const std::string mtx_filepath = "matrices/fidapm05.mtx";
    schwarz2lvl::SparseMatrixWrapper A;

    if (!schwarz2lvl::MatrixMarketIO::readMatrix(mtx_filepath, A)) {
        return EXIT_FAILURE;
    }

    // Simuliamo di voler dividere la matrice in 4 sottodomini
    int virtual_processors = 4;
    schwarz2lvl::GraphPartitioner partitioner;
    
    try {
        std::vector<int> part_map = partitioner.computePartition(A, virtual_processors);
        std::cout << "Partition successfully computed! Map size: " << part_map.size() << std::endl;
        
        // Stampiamo la destinazione delle prime 10 righe come verifica
        for(int i = 0; i < 10; ++i) {
            std::cout << "Row " << i << " is assigned to rank: " << part_map[i] << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
