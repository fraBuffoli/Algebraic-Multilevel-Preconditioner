#include "local_eigensolver.hpp"
#include <Eigen/Eigenvalues>
#include <iostream>
#include <vector>
#include <stdexcept>

namespace schwarz2lvl {

void LocalEigensolver::computeEigenpairs(const MatrixType& A_ii,
                                         const MatrixType& A_tilde_ii,
                                         const PartitionOfUnity& pou) {
    const Eigen::Index n_i = A_ii.rows();
    Z_i_.resize(n_i, 0); // Clear past state

    if (n_i == 0) return;

    // 1. Convert sparse matrices to dense matrices as required by Eigen's GeneralizedEigenSolver
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A_ii);
    Eigen::MatrixXd A_tilde_dense = Eigen::MatrixXd(A_tilde_ii);
    
    // 2. Apply the partition of unity weights to the local matrix A_ii
    const auto& weights = pou.getWeights();
    for (Eigen::Index r = 0; r < n_i; ++r) {
        for (Eigen::Index c = 0; c < n_i; ++c) {
            A_dense(r, c) = weights(r) * A_dense(r, c) * weights(c);
        }
    }

    // 3. Run the generalized eigensolver: A * z = lambda * B * z
    Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> ges;
    ges.compute(A_dense, A_tilde_dense);

    if (ges.info() != Eigen::Success) {
        throw std::runtime_error("LocalEigensolver Error: Generalized eigenvalue computation failed.");
    }

    // Internally, the problem is rewritten as beta*A*z = alpha*B*z, where lambda = alpha/beta.
    // This avoids division by zero and handles infinite eigenvalues correctly.
    const auto& alphas = ges.alphas();
    const auto& betas = ges.betas();
    const auto& eigenvectors = ges.eigenvectors();

    // 4. Scan eigenvalues and filter eigenvectors based on the criteria |lambda| > 1/tau
    std::vector<Eigen::Index> selected_indices;
    selected_indices.reserve(n_i);

    for (Eigen::Index i = 0; i < n_i; ++i) {
        // alphas(i) is complex, betas(i) is a real double
        double alpha = alphas(i).real();
        double beta = betas(i);
        
        // Check if eigenvalue is finite or infinite (avoiding divisions by zero)
        if (std::abs(beta) > 1e-14) {
            double lambda = alpha / beta;
            
            // If the eigenvalue exceeds the threshold tau, we keep it
            if (std::abs(lambda) >= 1.0 / tau_) {
                selected_indices.push_back(i);
            }
        } else {
            // If beta is almost zero, lambda tends to infinity.
            // Infinite eigenvalues are considered > 1/tau and should be included.
            selected_indices.push_back(i);
        }
    }

    // 5. Assemble the local space base matrix Z_i by extracting the selected columns
    Eigen::Index num_chosen = static_cast<Eigen::Index>(selected_indices.size());
    Z_i_.resize(n_i, num_chosen);

    for (Eigen::Index col = 0; col < num_chosen; ++col) {
        Eigen::Index original_idx = selected_indices[col];
        // Extract the real part of the eigenvector column
        Z_i_.col(col) = eigenvectors.col(original_idx).real();
    }
}

} // namespace schwarz2lvl
