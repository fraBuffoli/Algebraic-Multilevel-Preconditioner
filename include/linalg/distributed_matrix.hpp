/**
 * @file distributed_matrix.hpp
 * @brief Row-distributed sparse matrix-vector product y = A x.
 */
#ifndef SCHWARZ2LVL_LINALG_DISTRIBUTED_MATRIX_HPP
#define SCHWARZ2LVL_LINALG_DISTRIBUTED_MATRIX_HPP

#include "subdomain.hpp"
#include "types.hpp"

namespace schwarz2lvl::linalg {

/**
 * @class DistributedMatrix
 * @brief Global operator A, distributed by rows.
 *
 * Every process stores A(Omega_I_i, Omega_i) (its owned rows, with local
 * column numbering), so a product only needs the ghost values of x, which
 * are obtained with one HaloExchange::forward() (the same pattern as
 * OpenFOAM's `lduMatrix::Amul` with `initMatrixInterfaces` /
 * `updateMatrixInterfaces`).
 */
class DistributedMatrix {
public:
    /// @brief Wraps the owned rows of subdomain @p sd (not copied).
    explicit DistributedMatrix(const part::Subdomain& sd) : sd_(sd), xloc_(core::Vec::Zero(sd.nLocal())) {}

    /**
     * @brief y = A x (collective).
     * @param x Owned entries of x (size n_I).
     * @param y Owned entries of y (size n_I), overwritten.
     */
    void apply(const core::Vec& x, core::Vec& y) const
    {
        xloc_.head(sd_.nOwned()) = x;
        sd_.halo().forward(xloc_);
        y.noalias() = sd_.Aowned() * xloc_;
    }

    /// @brief Number of owned rows.
    core::LocalIndex nOwned() const { return sd_.nOwned(); }
    /// @brief Communicator.
    MPI_Comm comm() const { return sd_.comm(); }
    /// @brief Underlying subdomain.
    const part::Subdomain& subdomain() const { return sd_; }

private:
    const part::Subdomain& sd_; ///< Subdomain holding the owned rows and the halo.
    mutable core::Vec xloc_;    ///< Work vector on Omega_i.
};

} // namespace schwarz2lvl

#endif // SCHWARZ2LVL_LINALG_DISTRIBUTED_MATRIX_HPP
