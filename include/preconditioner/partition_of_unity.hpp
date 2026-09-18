#ifndef PARTITION_OF_UNITY_HPP
#define PARTITION_OF_UNITY_HPP

#include "sparse_matrix.hpp"
#include "subdomain_topology.hpp"
#include <vector>

namespace schwarz2lvl {

/**
 * @class PartitionOfUnity
 * @brief Computes and stores the diagonal partition of unity weights (D_i) for a subdomain.
 * 
 * This class ensures that corrections in overlapping zones sum up to the identity
 * globally by weighting nodes based on how many subdomains share them.
 */
class PartitionOfUnity {
public:
    /**
     * @brief Default constructor. Creates empty partition weights.
     */
    PartitionOfUnity() = default;

    /**
     * @brief Computes the diagonal weights based on the global partition map and subdomain topology.
     * 
     * @param matrix The global sparse matrix A.
     * @param partition_map The global mapping array from METIS (size n).
     * @param topology The computed topology of the target subdomain.
     */
    void computeWeights(const SparseMatrixWrapper& matrix,
                        const std::vector<int>& partition_map,
                        const SubdomainTopology& topology);

    /**
     * @brief Gets the computed diagonal weights vector.
     * @return Const reference to the Eigen vector containing the weights.
     */
    const VectorType& getWeights() const { return weights_; }

    /**
     * @brief Sets externally provided weights (e.g. received via MPI_Bcast from the subdomain that actually owns them).
     * @param w Raw weight values, ordered consistently with the subdomain's global index list.
     */
    void setWeights(const std::vector<double>& w) {
        weights_ = Eigen::Map<const VectorType>(
            w.data(), static_cast<Eigen::Index>(w.size()));
    }

private:
    VectorType weights_; // Local diagonal matrix entries (size n_i)
};

} // namespace schwarz2lvl

#endif // PARTITION_OF_UNITY_HPP
