#ifndef DISTRIBUTED_ONE_LEVEL_PRECONDITIONER_HPP
#define DISTRIBUTED_ONE_LEVEL_PRECONDITIONER_HPP

#include "config.hpp"
#include "local_matrix.hpp"
#include "halo_exchange.hpp"
#include "distributed_preconditioner.hpp"
#include <Eigen/SparseLU>

namespace schwarz2lvl {
/**
 * @class DistributedOneLevelPreconditioner
 * @brief Implements a distributed version of the classical restricted/additive Schwarz one-level preconditioner (ASM/RAS) for parallel computations.
 */
class DistributedOneLevelPreconditioner : public DistributedPreconditioner {
public:
    /**
     * @brief Default constructor creating an uninitialized distributed preconditioner.
     */
    DistributedOneLevelPreconditioner() = default;

    /**
     * @brief setup the distributed preconditioner by extracting and factorizing the local subdomain matrix A_ii.
     * @param local_A The local matrix A_ii for this subdomain.
     * @param halo The HaloExchange object for communication with neighboring subdomains.
     * @param pou_weights The partition of unity weights D_i for this subdomain.
     */
    void setup(const LocalMatrix& local_A, const HaloExchange& halo, const VectorType& pou_weights);

    /**
     * @brief Applies the distributed 1-level Schwarz preconditioner: z = M_ASM^-1 * r.
     * @param r_owned The local portion of the global residual vector.
     * @param z_owned The local portion of the global correction vector (accumulated).
     */
    void apply(const VectorType& r_owned, VectorType& z_owned) const override;

private:
    mutable Eigen::SparseLU<MatrixType> solver_;
    const HaloExchange* halo_ = nullptr;
    VectorType pou_weights_;
    Eigen::Index num_interior_ = 0;
    Eigen::Index n_i_ = 0;
    mutable VectorType r_ext_;
    mutable VectorType y_ext_;
};

} // namespace schwarz2lvl

#endif // DISTRIBUTED_ONE_LEVEL_PRECONDITIONER_HPP