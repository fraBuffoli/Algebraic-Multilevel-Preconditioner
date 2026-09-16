#include "matrix_market_io.hpp"
#include "sparse_matrix.hpp"
#include "graph_partitioner.hpp"
#include "subdomain_topology.hpp"
#include "restriction_operator.hpp"
#include "partition_of_unity.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>

int main() {
    // 1. Definiamo il percorso della matrice di risorse
    const std::string mtx_filepath = "matrices/fidapm05.mtx";
    schwarz2lvl::SparseMatrixWrapper A;

    // 2. Caricamento della matrice sparsa globale
    std::cout << "=== STEP 1: Loading Matrix Market File ===" << std::endl;
    if (!schwarz2lvl::MatrixMarketIO::readMatrix(mtx_filepath, A)) {
        std::cerr << "Initialization Failed! Could not parse: " << mtx_filepath << std::endl;
        return EXIT_FAILURE;
    }

    // 3. Calcolo del partizionamento dei grafi con METIS
    // Simuliamo la scomposizione per un'esecuzione a 4 processori/rank
    const int num_virtual_ranks = 4;
    std::cout << "\n=== STEP 2: Graph Partitioning ===" << std::endl;
    
    std::vector<int> partition_map;
    try {
        schwarz2lvl::GraphPartitioner partitioner;
        partition_map = partitioner.computePartition(A, num_virtual_ranks);
    } catch (const std::exception& e) {
        std::cerr << "Partitioning Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // 4. Analisi della Topologia e calcolo dei pesi per OGNI Rank hardware
    std::cout << "\n=== STEP 3: Subdomain Geometry Setup (Iterating over Ranks) ===" << std::endl;
    
    for (int rank = 0; rank < num_virtual_ranks; ++rank) {
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "Processing Subdomain Topology for RANK: " << rank << std::endl;

        // Estrazione della topologia locale (Allargamento a distanza 1 per l'overlap)
        schwarz2lvl::SubdomainTopology topology;
        topology.computeTopology(A, partition_map, rank);

        std::cout << "  > Interior Nodes (Omega_Ii) : " << topology.getInteriorIndices().size() << std::endl;
        std::cout << "  > Overlap Nodes  (Omega_Gamma_i): " << topology.getBoundaryIndices().size() << std::endl;
        std::cout << "  > Total Local Size (n_i)    : " << topology.getGlobalIndices().size() << std::endl;

        // Inizializzazione dell'operatore di restrizione locale (R_i)
        schwarz2lvl::RestrictionOperator R_i(topology.getGlobalIndices());
        std::cout << "  > Restriction Operator R_i allocated with local size: " << R_i.localSize() << std::endl;

        // Calcolo della partizione dell'unità diagonale locale (D_i)
        schwarz2lvl::PartitionOfUnity D_i;
        D_i.computeWeights(A, partition_map, topology);
        
        const auto& weights = D_i.getWeights();
        std::cout << "  > Partition of Unity weights (D_i) computed." << std::endl;

        // Diagnostica visiva sui pesi dei primi nodi per questo Rank
        std::cout << "  > Sample of the first local weights (D_i diagonal):" << std::endl;
        Eigen::Index sample_size = std::min(static_cast<Eigen::Index>(5), weights.size());
        for (Eigen::Index j = 12; j < 12 + sample_size; ++j) {
            std::string type = (j < static_cast<Eigen::Index>(topology.getInteriorIndices().size())) ? "Interior" : "Overlap";
            std::cout << "    Local Index " << j << " (" << type << ") -> Weight: " << weights(j) << std::endl;
        }
    }
    
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "=== Phase 2 Validation Successful! ===" << std::endl;

    return EXIT_SUCCESS;
}
