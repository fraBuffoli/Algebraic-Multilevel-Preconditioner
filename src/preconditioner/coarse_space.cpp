#include "coarse_space.hpp"
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
    const size_t num_subdomains = local_Z.size();

    // 1. Calculate the total dimension of the Coarse Space by summing chosen modes from all ranks
    total_coarse_dim_ = 0;
    for (size_t i = 0; i < num_subdomains; ++i) {
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

    // 2. Build the global restriction operator R_0 row by row
    // R_0 consists of stacked blocks of (R_i^T * D_i * Z_i)^T = Z_i^T * D_i * R_i
    R_0_ = Eigen::MatrixXd::Zero(total_coarse_dim_, n);
    
    Eigen::Index current_coarse_row = 0;
    for (size_t i = 0; i < num_subdomains; ++i) {
        const auto& Z_i = local_Z[i];
        const auto& R_i = restrictions[i];
        const auto& D_i = pous[i].getWeights();
        const auto& local_indices = R_i.getGlobalIndices();
        
        Eigen::Index local_modes = Z_i.cols();
        Eigen::Index n_i = Z_i.rows();

        // Map the columns of Z_i directly into the global slots of R_0 using the global topology indices
        for (Eigen::Index mode = 0; mode < local_modes; ++mode) {
            for (Eigen::Index j = 0; j < n_i; ++j) {
                int global_col = local_indices[j];
                // Apply the combination: component of eigenvector * partition of unity weight
                R_0_(current_coarse_row + mode, global_col) = Z_i(j, mode) * D_i(j);
            }
        }
        current_coarse_row += local_modes;
    }

    // 3. Assemble the reduced global coarse grid matrix A_00 = R_0 * A * R_0^T
    std::cout << "Computing reduced system matrix A_00 = R_0 * A * R_0^T..." << std::endl;
    const Eigen::MatrixXd R0t = R_0_.transpose();
    const Eigen::MatrixXd A_R0t = global_A.getMatrix() * R0t;   // sparse * dense
    const Eigen::MatrixXd A_00 = R_0_ * A_R0t;

    // 4. Pre-factorize the coarse system A_00 using dense Partial-Pivoting LU
    A_00_lu_.compute(A_00);
    
    const Eigen::VectorXd diagU = A_00_lu_.matrixLU().diagonal().cwiseAbs();
    if (diagU.minCoeff() < 1e-12 * diagU.maxCoeff()) {
        throw std::runtime_error("CoarseSpace Error: A_00 is numerically singular (smallest LU pivot below relative threshold).");
    }
    std::cout << "Coarse Space operator successfully factorized." << std::endl;
}

void CoarseSpace::apply(const VectorType& r, VectorType& z) const {

    if (total_coarse_dim_ == 0) return;

    // Step A: Restrict global residual to coarse space -> r_coarse = R_0 * r
    Eigen::VectorXd r_coarse = R_0_ * r;

    // Step B: Solve the reduced linear system -> y_coarse = A_00^-1 * r_coarse
    Eigen::VectorXd y_coarse = A_00_lu_.solve(r_coarse);

    // Step C: Prolong coarse correction to global size and accumulate -> z = z + R_0^T * y_coarse
    z += R_0_.transpose() * y_coarse;
}

} // namespace schwarz2lvl
