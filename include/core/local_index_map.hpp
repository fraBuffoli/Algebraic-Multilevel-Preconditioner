#ifndef LOCAL_INDEX_MAP_HPP
#define LOCAL_INDEX_MAP_HPP

#include "config.hpp"
#include <vector>
#include <unordered_map>

namespace schwarz2lvl {

/**
 * @class LocalIndexMap
 * @brief Map between local and global indices for a subdomain in a parallel computation.
 */
class LocalIndexMap {
public:
    /**
     * @brief Default constructor for LocalIndexMap.
     * Initializes an empty LocalIndexMap.
     */
    LocalIndexMap() = default;

    /**
     * @brief Build the local index map from a list of global indices.
     * @param extended_global_indices A vector of global indices for the local subdomain, with interior indices first, followed by boundary indices.
     * @param num_interior The number of interior indices in the extended_global_indices vector. The remaining indices are considered boundary indices. 
     */
    void build(const std::vector<int>& extended_global_indices, Eigen::Index num_interior);

    Eigen::Index size() const { return static_cast<Eigen::Index>(global_indices_.size()); }
    Eigen::Index numInterior() const { return num_interior_; }
    Eigen::Index numBoundary() const { return size() - num_interior_; }

    /** 
     * @brief Get the global index corresponding to a local position.
     * @param local_pos The local position (index) in the subdomain.
     * @return The corresponding global index.
     */
    int localToGlobal(Eigen::Index local_pos) const { return global_indices_[local_pos]; }

    /**
     * @brief Get the local position corresponding to a global index.
     * @param global_id The global index.
     * @return The corresponding local position, or -1 if the global index is not in the local subdomain.
     */
    int globalToLocal(int global_id) const;

    bool isInterior(Eigen::Index local_pos) const { return local_pos < num_interior_; }

    const std::vector<int>& globalIndices() const { return global_indices_; }

private:
    std::vector<int> global_indices_;              
    std::unordered_map<int, int> global_to_local_;
    Eigen::Index num_interior_ = 0;
};

} // namespace schwarz2lvl
#endif // LOCAL_INDEX_MAP_HPP
