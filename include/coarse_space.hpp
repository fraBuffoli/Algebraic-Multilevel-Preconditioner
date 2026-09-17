#ifndef COARSE_SPACE_HPP
#define COARSE_SPACE_HPP

#include "sparse_matrix.hpp"
#include "restriction_operator.hpp"
#include "partition_of_unity.hpp"
#include <Eigen/Core>
#include <Eigen/LU>
#include <vector>

namespace schwarz2lvl {

/**
 * @class CoarseSpace
 * @brief Assembles and manages the global spectral coarse grid operator (A_00).
 * 
 * This class collects all local eigenvector bases Z_i, constructs the global 
 * projection matrix R_0, and factorizes the reduced system matrix A_00 = R_0 * A * R_0^T.
 */
class CoarseSpace {
public:
    /**
     * @brief Default constructor creating an uninitialized coarse space.
     */
    CoarseSpace() = default;

    /**
     * @brief Builds and factorizes the global coarse matrix A_00 by looping over subdomains.
     * 
     * @param global_A The global sparse matrix wrapper.
     * @param local_Z A vector containing the filtered local eigenvector bases Z_i for all subdomains.
     * @param restrictions A vector containing the restriction operators R_i for all subdomains.
     * @param pous A vector containing the partition of unity weights D_i for all subdomains.
     */
    void setup(const SparseMatrixWrapper& global_A,
               const std::vector<Eigen::MatrixXd>& local_Z,
               const std::vector<RestrictionOperator>& restrictions,
               const std::vector<PartitionOfUnity>& pous);

    /**
     * @brief Solves the coarse space system: y = R_0^T * A_00^-1 * R_0 * r
     * 
     * Applies the coarse correction projection onto a global residual vector.
     * 
     * @param r Input global residual vector.
     * @param z Output global correction vector where the coarse space update is accumulated (+=).
     */
    void apply(const VectorType& r,
               VectorType& z) const;

    /**
     * @brief Gets the total dimension of the global Coarse Space.
     * @return Number of global spectral modes.
     */
    Eigen::Index coarseSize() const { return A_00_lu_.rows(); }

private:
    Eigen::Index total_coarse_dim_ = 0; // Total number of selected global modes
    Eigen::MatrixXd R_0_; // Global restriction matrix R_0 (size total_coarse_dim x n)
    Eigen::PartialPivLU<Eigen::MatrixXd> A_00_lu_; // Dense LU solver for the small coarse system A_00
};

} // namespace schwarz2lvl

#endif // COARSE_SPACE_HPP
