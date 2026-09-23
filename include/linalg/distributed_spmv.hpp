#ifndef DISTRIBUTED_SPMV_HPP
#define DISTRIBUTED_SPMV_HPP

#include "config.hpp"
#include "local_matrix.hpp"
#include "halo_exchange.hpp"

namespace schwarz2lvl {

/**
 * @class DistributedSpMV
 * @brief Class for performing distributed sparse matrix-vector multiplication (SpMV).
 */
class DistributedSpMV {
public:
    /**
     * @brief Default constructor for DistributedSpMV.
     * Initializes an empty DistributedSpMV object.
     */
    DistributedSpMV() = default;

    /**
     * @brief Sets up the DistributedSpMV object with the local matrix and halo exchange information.
     * @param local_A The LocalMatrix representing the local portion of the sparse matrix.
     * @param halo The HaloExchange object containing the communication pattern for exchanging boundary data.
     */
    void setup(const LocalMatrix& local_A, const HaloExchange& halo);

    /**
     * @brief Applies the distributed sparse matrix-vector multiplication.
     * @param x_owned The local vector containing the owned values (interior).
     * @param y_owned The local vector to store the result of the multiplication (interior).
     */
    void apply(const VectorType& x_owned, VectorType& y_owned) const;

    /**
     * @brief Returns the number of interior (owned) elements in the local vector.
     * @return The number of interior elements.
     */
    Eigen::Index numInterior() const { return num_interior_; }

private:
    MatrixType A_interior_local_;   
    const HaloExchange* halo_ = nullptr;
    Eigen::Index num_interior_ = 0;
    Eigen::Index n_i_ = 0;
    mutable VectorType x_ext_;  
};

} // namespace schwarz2lvl

#endif // DISTRIBUTED_SPMV_HPP