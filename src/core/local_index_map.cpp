#include "local_index_map.hpp"

namespace schwarz2lvl {

void LocalIndexMap::build(const std::vector<int>& extended_global_indices, Eigen::Index num_interior) {
    global_indices_ = extended_global_indices;
    num_interior_ = num_interior;

    global_to_local_.clear();
    global_to_local_.reserve(global_indices_.size() * 2);
    for (size_t j = 0; j < global_indices_.size(); ++j) {
        global_to_local_[global_indices_[j]] = static_cast<int>(j);
    }
}

int LocalIndexMap::globalToLocal(int global_id) const {
    const auto it = global_to_local_.find(global_id);
    return (it != global_to_local_.end()) ? it->second : -1;
}

} // namespace schwarz2lvl
