#ifndef DISTRIBUTED_COARSE_SPACE_HPP
#define DISTRIBUTED_COARSE_SPACE_HPP

#include "config.hpp"
#include "halo_exchange.hpp"
#include "distributed_spmv.hpp"
#include <Eigen/Core>
#include <Eigen/LU>
#include <vector>

namespace schwarz2lvl {

/**
 * @class DistributedCoarseSpace
 * @brief Implements a distributed coarse space for two-level Schwarz methods.
 */
class DistributedCoarseSpace {
public:
    /**
     * @brief Default constructor for DistributedCoarseSpace.
     * Initializes an empty DistributedCoarseSpace object.
     */
    DistributedCoarseSpace() = default;

    /**
     * @brief Sets up the distributed coarse space with the provided halo exchange, distributed SpMV, partition of unity weights, and local basis functions.
     * @param halo The HaloExchange object for boundary data exchange.
     * @param spmv The DistributedSpMV object for applying the local matrix.
     * @param pou_weights The partition of unity weights (D_i) for the local subdomain.
     * @param Z_i The local basis functions (n_i x m_i) for the subdomain.
     */
    void setup(const HaloExchange& halo,
               const DistributedSpMV& spmv,
               const VectorType& pou_weights,
               const Eigen::MatrixXd& Z_i);

    /**
     * @brief Applies the distributed coarse space correction to the input vector.
     * @param r_owned The local vector containing the owned values (interior).
     * @param z_owned The local vector to store the result of the coarse space correction (interior).
     */
    void apply(const VectorType& r_owned, VectorType& z_owned) const;

    bool isEmpty() const { return total_coarse_dim_ == 0; }
    Eigen::Index coarseSize() const { return total_coarse_dim_; }
    Eigen::Index myCoarseOffset() const { return my_coarse_offset_; }
    Eigen::Index myNumModes() const { return my_num_modes_; }
    const std::vector<int>& modesPerRank() const { return modes_per_rank_; }
    const Eigen::MatrixXd& coarseMatrix() const { return A_00_; }

private:
    const HaloExchange* halo_ = nullptr;
    Eigen::MatrixXd W_;                  
    Eigen::Index num_interior_ = 0;
    Eigen::Index n_i_ = 0;

    Eigen::Index total_coarse_dim_ = 0;  
    Eigen::Index my_coarse_offset_ = 0;
    Eigen::Index my_num_modes_ = 0;     
    std::vector<int> modes_per_rank_;
    std::vector<int> mode_offsets_;

    Eigen::MatrixXd A_00_;
    Eigen::PartialPivLU<Eigen::MatrixXd> A_00_lu_;

    mutable VectorType r_ext_;
    mutable VectorType r_coarse_local_;
    mutable VectorType r_coarse_global_;
    mutable VectorType z_ext_;
};

} // namespace schwarz2lvl

#endif // DISTRIBUTED_COARSE_SPACE_HPP