#include "distributed_coarse_space.hpp"
#include <mpi.h>
#include <iostream>
#include <stdexcept>

namespace schwarz2lvl {

void DistributedCoarseSpace::setup(const HaloExchange& halo,
                                   const DistributedSpMV& spmv,
                                   const VectorType& pou_weights,
                                   const Eigen::MatrixXd& Z_i) {
    int my_rank = 0, num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    halo_ = &halo;
    num_interior_ = halo.numInterior();
    n_i_ = pou_weights.size();
    if (Z_i.rows() != n_i_) {
        throw std::invalid_argument("DistributedCoarseSpace::setup: Z_i.rows() != n_i.");
    }
    if (spmv.numInterior() != num_interior_) {
        throw std::invalid_argument("DistributedCoarseSpace::setup: spmv e halo incoerenti.");
    }

    W_ = pou_weights.asDiagonal() * Z_i;
    my_num_modes_ = Z_i.cols();

    const int my_modes_int = static_cast<int>(my_num_modes_);
    modes_per_rank_.assign(num_ranks, 0);
    MPI_Allgather(&my_modes_int, 1, MPI_INT, modes_per_rank_.data(), 1, MPI_INT, MPI_COMM_WORLD);
    mode_offsets_.assign(num_ranks, 0);
    total_coarse_dim_ = 0;
    for (int r = 0; r < num_ranks; ++r) {
        mode_offsets_[r] = static_cast<int>(total_coarse_dim_);
        total_coarse_dim_ += modes_per_rank_[r];
    }
    my_coarse_offset_ = mode_offsets_[my_rank];

    r_ext_.resize(n_i_);
    z_ext_.resize(n_i_);
    r_coarse_local_.resize(my_num_modes_);
    r_coarse_global_.resize(total_coarse_dim_);

    if (total_coarse_dim_ == 0) {
        if (my_rank == 0) {
            std::cout << "DistributedCoarseSpace: nessun modo selezionato, correzione "
                         "grossolana disattivata." << std::endl;
        }
        return;
    }

    Eigen::MatrixXd my_cols(total_coarse_dim_, my_num_modes_);
    VectorType ext_basis(n_i_), phi_owned, y_owned, y_ext(n_i_);

    for (int j = 0; j < num_ranks; ++j) {
        for (int b = 0; b < modes_per_rank_[j]; ++b) {
            const Eigen::Index c = mode_offsets_[j] + b;

            if (my_rank == j) ext_basis = W_.col(b);
            else              ext_basis.setZero();
            halo_->scatterAddToOwner(ext_basis, phi_owned);   // a)

            spmv.apply(phi_owned, y_owned);                    // b)

            y_ext.head(num_interior_) = y_owned;               // c)
            halo_->exchange(y_ext);

            if (my_num_modes_ > 0) {                           // d)
                my_cols.row(c).noalias() = (W_.transpose() * y_ext).transpose();
            }
        }
    }

    std::vector<int> counts(num_ranks), displs(num_ranks);
    for (int r = 0; r < num_ranks; ++r) {
        counts[r] = modes_per_rank_[r] * static_cast<int>(total_coarse_dim_);
        displs[r] = mode_offsets_[r] * static_cast<int>(total_coarse_dim_);
    }
    Eigen::MatrixXd A00T(total_coarse_dim_, total_coarse_dim_);
    MPI_Allgatherv(my_cols.data(), counts[my_rank], MPI_DOUBLE,
                   A00T.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    A_00_ = A00T.transpose();

    A_00_lu_.compute(A_00_);
    const Eigen::VectorXd diagU = A_00_lu_.matrixLU().diagonal().cwiseAbs();
    if (diagU.minCoeff() < 1e-12 * diagU.maxCoeff()) {
        throw std::runtime_error("DistributedCoarseSpace Error: A_00 numericamente singolare "
                                 "(pivot LU minimo sotto la soglia relativa).");
    }
}

void DistributedCoarseSpace::apply(const VectorType& r_owned, VectorType& z_owned) const {
    if (total_coarse_dim_ == 0) {
        z_owned.setZero(num_interior_);
        return;
    }

    r_ext_.head(num_interior_) = r_owned;
    halo_->exchange(r_ext_);

    r_coarse_local_.noalias() = W_.transpose() * r_ext_;

    MPI_Allgatherv(r_coarse_local_.data(), static_cast<int>(my_num_modes_), MPI_DOUBLE,
                   r_coarse_global_.data(), modes_per_rank_.data(), mode_offsets_.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    const VectorType y_coarse = A_00_lu_.solve(r_coarse_global_);

    if (my_num_modes_ > 0) z_ext_.noalias() = W_ * y_coarse.segment(my_coarse_offset_, my_num_modes_);
    else                   z_ext_.setZero();

    halo_->scatterAddToOwner(z_ext_, z_owned);
}

} // namespace schwarz2lvl