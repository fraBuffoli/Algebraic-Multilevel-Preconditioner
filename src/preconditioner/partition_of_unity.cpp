#include "partition_of_unity.hpp"
#include <algorithm>
#include <set>

namespace schwarz2lvl {

void PartitionOfUnity::computeWeights(const SparseMatrixWrapper& matrix,
                                      const std::vector<int>& partition_map,
                                      const SubdomainTopology& topology) {

    const Eigen::Index n = matrix.rows();
    const auto& eigen_mat = matrix.getMatrix();
    const auto& local_indices = topology.getGlobalIndices();
    const Eigen::Index n_i = static_cast<Eigen::Index>(local_indices.size());

    std::vector<int> global_to_local(n, -1);
    for (Eigen::Index j = 0; j < n_i; ++j) {
        global_to_local[local_indices[j]] = static_cast<int>(j);
    }

    // Stessa definizione di multiplicity dell'originale, in UN solo passaggio
    // O(nnz(A)) invece di N passaggi O(n) ciascuno.
    std::vector<std::set<int>> reached_by(n_i);

    for (Eigen::Index u = 0; u < n; ++u) {
        const int r = partition_map[u];
        for (MatrixType::InnerIterator it(eigen_mat, u); it; ++it) {
            const int v = static_cast<int>(it.col());
            const int local_j = global_to_local[v];
            if (local_j == -1) continue;
            if (partition_map[v] == r) continue;
            reached_by[local_j].insert(r);
        }
    }

    weights_.resize(n_i);
    for (Eigen::Index j = 0; j < n_i; ++j) {
        const int multiplicity = 1 + static_cast<int>(reached_by[j].size());
        weights_(j) = 1.0 / static_cast<double>(multiplicity);
    }
}

} // namespace schwarz2lvl