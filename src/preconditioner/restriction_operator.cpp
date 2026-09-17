#include "restriction_operator.hpp"
#include <stdexcept>

namespace schwarz2lvl {

void RestrictionOperator::apply(const VectorType& g_vec, VectorType& l_vec) const {
    const Eigen::Index n_i = localSize();
    l_vec.resize(n_i);

    for (Eigen::Index j = 0; j < n_i; ++j) {
        // Safety check: prevent out-of-bounds reading from the global vector
        if (global_indices_[j] >= g_vec.size() || global_indices_[j] < 0) {
            throw std::out_of_range("Restriction Error: Mapped global index is out of bounds for the input vector.");
        }
        l_vec(j) = g_vec(global_indices_[j]);
    }
}

void RestrictionOperator::applyTranspose(const VectorType& l_vec, VectorType& g_vec) const {
    const Eigen::Index n_i = localSize();

    for (Eigen::Index j = 0; j < n_i; ++j) {
        // Safety check: prevent out-of-bounds writing into the global vector
        if (global_indices_[j] >= g_vec.size() || global_indices_[j] < 0) {
            throw std::out_of_range("Prolongation Error: Mapped global index is out of bounds for the output vector.");
        }
        g_vec(global_indices_[j]) += l_vec(j);
    }
}

} // namespace schwarz2lvl
