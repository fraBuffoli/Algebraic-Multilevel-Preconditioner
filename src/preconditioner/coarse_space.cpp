#include "coarse_space.hpp"
#include <Eigen/Sparse>
#include <iostream>
#include <stdexcept>
#include <mpi.h>

namespace schwarz2lvl {

void CoarseSpace::setup(const SparseMatrixWrapper& global_A,
                        const std::vector<Eigen::MatrixXd>& local_Z,
                        const std::vector<RestrictionOperator>& restrictions,
                        const std::vector<PartitionOfUnity>& pous) {

    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    const Eigen::Index n = global_A.rows();
    n_global_ = n;
    const size_t num_subdomains = local_Z.size();

    total_coarse_dim_ = 0;
    modes_per_rank_.assign(num_subdomains, 0);
    mode_offsets_.assign(num_subdomains, 0);
    for (size_t i = 0; i < num_subdomains; ++i) {
        modes_per_rank_[i] = static_cast<int>(local_Z[i].cols());
        mode_offsets_[i]   = static_cast<int>(total_coarse_dim_);
        total_coarse_dim_ += local_Z[i].cols();
    }

    std::cout << "Assembling Coarse Space projection matrix R_0. Total dimensions: "
              << total_coarse_dim_ << " x " << n << std::endl;

    if (total_coarse_dim_ == 0) {
        if (my_rank == 0) {
            std::cout << "CoarseSpace: no eigenvalue exceeded the threshold 1/tau. "
                         "The coarse correction is disabled and the preconditioner "
                         "degrades to one-level Schwarz. This is expected for N = 1, "
                         "where the local solve is already exact." << std::endl;
        }
        return;
    }

    my_restriction_    = restrictions[my_rank];
    my_pou_             = pous[my_rank];
    my_Z_               = local_Z[my_rank];
    my_num_modes_       = my_Z_.cols();
    my_coarse_offset_   = mode_offsets_[my_rank];


    std::vector<Eigen::Triplet<double>> r0_triplets;
    Eigen::Index nnz_estimate = 0;
    for (size_t i = 0; i < num_subdomains; ++i) {
        nnz_estimate += local_Z[i].rows() * local_Z[i].cols();
    }
    r0_triplets.reserve(static_cast<size_t>(nnz_estimate));

    Eigen::Index current_coarse_row = 0;
    for (size_t i = 0; i < num_subdomains; ++i) {
        const auto& Z_i = local_Z[i];
        const auto& R_i = restrictions[i];
        const auto& D_i = pous[i].getWeights();
        const auto& local_indices = R_i.getGlobalIndices();

        Eigen::Index local_modes = Z_i.cols();
        Eigen::Index n_i = Z_i.rows();

        for (Eigen::Index mode = 0; mode < local_modes; ++mode) {
            for (Eigen::Index j = 0; j < n_i; ++j) {
                const double val = Z_i(j, mode) * D_i(j);
                if (val == 0.0) continue;
                int global_col = local_indices[j];
                r0_triplets.emplace_back(current_coarse_row + mode, global_col, val);
            }
        }
        current_coarse_row += local_modes;
    }

    Eigen::SparseMatrix<double> R_0(total_coarse_dim_, n);
    R_0.setFromTriplets(r0_triplets.begin(), r0_triplets.end());
    R_0.makeCompressed();

    std::cout << "Computing reduced system matrix A_00 = R_0 * A * R_0^T..." << std::endl;
    const Eigen::SparseMatrix<double> R0t = R_0.transpose();
    const Eigen::SparseMatrix<double> A_R0t = global_A.getMatrix() * R0t;
    const Eigen::MatrixXd A_00 = Eigen::MatrixXd(R_0 * A_R0t);

    A_00_lu_.compute(A_00);

    const Eigen::VectorXd diagU = A_00_lu_.matrixLU().diagonal().cwiseAbs();
    if (diagU.minCoeff() < 1e-12 * diagU.maxCoeff()) {
        throw std::runtime_error("CoarseSpace Error: A_00 is numerically singular (smallest LU pivot below relative threshold).");
    }
    std::cout << "Coarse Space operator successfully factorized." << std::endl;

    r_local_.resize(my_restriction_.localSize());
    r_coarse_local_.resize(my_num_modes_);
    r_coarse_global_.resize(total_coarse_dim_);
    z_local_.resize(my_restriction_.localSize());
    z_this_.resize(n_global_);
    z_sum_.resize(n_global_);
}

void CoarseSpace::apply(const VectorType& r, VectorType& z) const {

    if (total_coarse_dim_ == 0) return;

    my_restriction_.apply(r, r_local_);

    r_coarse_local_.noalias() = my_Z_.transpose() * r_local_.cwiseProduct(my_pou_.getWeights());

    MPI_Allgatherv(r_coarse_local_.data(), static_cast<int>(my_num_modes_), MPI_DOUBLE,
                   r_coarse_global_.data(), modes_per_rank_.data(), mode_offsets_.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    Eigen::VectorXd y_coarse = A_00_lu_.solve(r_coarse_global_);

    z_local_.noalias() = my_pou_.getWeights().cwiseProduct
    (
        my_Z_ * y_coarse.segment(my_coarse_offset_, my_num_modes_)
    );

    z_this_.setZero();
    my_restriction_.applyTranspose(z_local_, z_this_);
    MPI_Allreduce(z_this_.data(), z_sum_.data(), static_cast<int>(n_global_),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    z += z_sum_;
}

} // namespace schwarz2lvl