#ifndef SUBDOMAIN_TOPOLOGY_HPP
#define SUBDOMAIN_TOPOLOGY_HPP

#include "global_matrix.hpp"
#include <vector>

namespace schwarz2lvl {

/**
 * @class SubdomainTopology
 * @brief Computes and stores the index sets defining a subdomain's algebraic topology.
 * 
 * This class takes the global partition map from METIS and expands the local 
 * non-overlapping nodes by a distance of 1 in the graph to construct the overlap.
 */
class SubdomainTopology {
public:
    /**
     * @brief Default constructor. Creates an empty topology.
     */
    SubdomainTopology() = default;

    /**
     * @brief Computes the interior, boundary, and total index sets for a specific rank.
     * 
     * @param matrix The global sparse matrix A.
     * @param partition_map The global mapping array from METIS (size n).
     * @param target_rank The MPI rank (subdomain ID) of the process executing the computation.
     */
    void computeTopology(const SparseMatrixWrapper& matrix, 
                         const std::vector<int>& partition_map, 
                         int target_rank);

    
    /**
     * @brief Gets the complete set of global indices (Interior + Overlap Boundary).
     * @return Const reference to the full local index vector.
     */
    const std::vector<int>& getGlobalIndices() const { return global_indices_; }

    /**
     * @brief Gets the indices of purely interior nodes (\f$\Omega_{Ii}\f$).
     * @return Const reference to the interior index vector.
     */
    const std::vector<int>& getInteriorIndices() const { return interior_indices_; }

    /**
     * @brief Gets the indices of overlapping boundary nodes (\f$\Omega_{\Gamma i}\f$).
     * @return Const reference to the boundary index vector.
     */
    const std::vector<int>& getBoundaryIndices() const { return boundary_indices_; }

private:
    std::vector<int> interior_indices_; // Purely interior nodes assigned by METIS (\f$\Omega_{Ii}\f$)
    std::vector<int> boundary_indices_; // Overlapping nodes belonging to other ranks (\f$\Omega_{\Gamma i}\f$)
    std::vector<int> global_indices_;   // Total concatenated list of local unknowns (\f$\Omega_i\f$)
};

} // namespace schwarz2lvl

#endif // SUBDOMAIN_TOPOLOGY_HPP
