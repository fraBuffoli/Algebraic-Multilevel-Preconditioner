#ifndef HALO_EXCHANGE_HPP
#define HALO_EXCHANGE_HPP

#include "config.hpp"
#include "subdomain_topology.hpp"
#include <vector>

namespace schwarz2lvl {

/**
 * @class HaloExchange
 * @brief Implements the neighbor-to-neighbor communication pattern for exchanging boundary data between subdomains.
 * 
 * Builds and executes the neighbor-to-neighbor communication pattern needed
 * whenever this rank needs, for its own EXTENDED (interior + boundary) index
 * set, up-to-date values at boundary nodes that some OTHER rank owns.
 * This is the primitive that lets a rank talk ONLY to the ranks it actually
 * shares nodes with, instead of relying on a full-length MPI_Allreduce
 */
class HaloExchange {
public:
    /**
     * @brief Default constructor. Creates an uninitialized HaloExchange object.
     */
    HaloExchange() = default;

    /**
     * @brief Initializes the HaloExchange object with the necessary communication pattern.
     *
     * One-time setup. Must be called with the SAME partition_map on every rank.
     */
    void setup(const std::vector<int>& partition_map,
               const SubdomainTopology& topology);

    /**
     * @brief Exchanges boundary data with neighboring subdomains.
     *
     * @param local_values The local vector containing both interior and boundary values.
     */
    void exchange(VectorType& local_values) const;

    /**
     * @brief Complementary/adjoint operation for scattering and adding values.
     *
     * @param local_values The local vector containing both interior and boundary values.
     * @param owned_result The vector to store the accumulated contributions.
     */
    void scatterAddToOwner(const VectorType& local_values, VectorType& owned_result) const;

    int numNeighbors() const { return static_cast<int>(neighbor_ranks_.size()); }
    const std::vector<int>& getNeighborRanks() const { return neighbor_ranks_; }
    Eigen::Index numInterior() const { return num_interior_; }

private:
    std::vector<int> neighbor_ranks_;
    std::vector<std::vector<int>> send_local_positions_;
    std::vector<std::vector<int>> recv_local_positions_;

    Eigen::Index num_interior_ = 0;
};

} // namespace schwarz2lvl
#endif // HALO_EXCHANGE_HPP