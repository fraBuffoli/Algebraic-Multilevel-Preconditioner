#include "partition_of_unity.hpp"
#include <algorithm>

namespace schwarz2lvl {

void PartitionOfUnity::computeWeights(const SparseMatrixWrapper& matrix,
                                      const std::vector<int>& partition_map,
                                      const SubdomainTopology& topology) {

    const Eigen::Index n = matrix.rows();
    const auto& eigen_mat = matrix.getMatrix();

    const int num_subdomains =
        *std::max_element(partition_map.begin(), partition_map.end()) + 1;

    // multiplicity[v] = number of overlapping subdomains Omega_i containing v.
    // Every node is contained at least by its own METIS owner (as an interior node).
    std::vector<int>  multiplicity(n, 1);
    std::vector<char> touched_by_r(n, 0);

    // v belongs to Omega_Gamma_r if some u with partition_map[u] == r has A(u,v) != 0.
    for (int r = 0; r < num_subdomains; ++r) {
        std::fill(touched_by_r.begin(), touched_by_r.end(), 0);
        for (Eigen::Index u = 0; u < n; ++u) {
            if (partition_map[u] != r) continue;
            for (MatrixType::InnerIterator it(eigen_mat, u); it; ++it) {
                const int v = static_cast<int>(it.col());
                if (partition_map[v] != r && !touched_by_r[v]) {
                    touched_by_r[v] = 1;
                    multiplicity[v]++;   // subdomain r contains v in its overlap
                }
            }
        }
    }

    const auto& local_indices = topology.getGlobalIndices();
    weights_.resize(static_cast<Eigen::Index>(local_indices.size()));
    for (Eigen::Index j = 0; j < weights_.size(); ++j) {
        weights_(j) = 1.0 / static_cast<double>(multiplicity[local_indices[j]]);
    }
}

} // namespace schwarz2lvl