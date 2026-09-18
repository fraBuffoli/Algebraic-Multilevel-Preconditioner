#include "local_eigensolver.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <complex>
#include <algorithm>

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

    // 4. Scan eigenvalues and filter eigenvectors based on the criterion |lambda| >= 1/tau.
    //    lambda = alpha / beta with alpha complex: use the complex modulus.
    std::vector<Eigen::Index> selected_indices;
    selected_indices.reserve(n_i);

    double alpha_scale = 0.0, beta_scale = 0.0;
    for (Eigen::Index i = 0; i < n_i; ++i) {
        alpha_scale = std::max(alpha_scale, std::abs(alphas(i)));
        beta_scale  = std::max(beta_scale,  std::abs(betas(i)));
    }
    const double beta_tol  = kInfiniteEigenvalueRelTol * std::max(beta_scale, 1.0);
    const double alpha_tol = kInfiniteEigenvalueRelTol * std::max(alpha_scale, 1.0);

    for (Eigen::Index i = 0; i < n_i; ++i) {
        const double abs_alpha = std::abs(alphas(i));  
        const double abs_beta  = std::abs(betas(i));

        if (abs_beta > beta_tol) {
            if (abs_alpha / abs_beta >= 1.0 / tau_) {
                selected_indices.push_back(i);
            }
        } else if (abs_alpha > alpha_tol) {
            // beta ~ 0, alpha != 0: genuinely infinite eigenvalue, i.e. a direction
            // in K_i = ker(A_tilde_ii) but not in L_i. Eq. (3.2) keeps these.
            selected_indices.push_back(i);
        }
        // else: alpha ~ 0 AND beta ~ 0 -> indeterminate 0/0 direction, lying in
        // L_i ∩ K_i. Eq. (3.2) explicitly excludes it from the coarse space.
    }

    // Keep at most nev_ modes, those with the largest |lambda| (paper, section 4).
    if (static_cast<Eigen::Index>(selected_indices.size()) > nev_) {
        std::sort(selected_indices.begin(), selected_indices.end(),
                  [&](Eigen::Index a, Eigen::Index b) {
                      const double la = std::abs(betas(a)) > beta_tol
                                      ? std::abs(alphas(a)) / std::abs(betas(a))
                                      : std::numeric_limits<double>::infinity();
                      const double lb = std::abs(betas(b)) > beta_tol
                                      ? std::abs(alphas(b)) / std::abs(betas(b))
                                      : std::numeric_limits<double>::infinity();
                      return la > lb;
                  });
        selected_indices.resize(nev_);
    }

    // 5. Assemble the local basis Z_i.
    Eigen::MatrixXd candidates(n_i, 2 * static_cast<Eigen::Index>(selected_indices.size()));
    Eigen::Index ncand = 0;

    for (Eigen::Index col = 0; col < static_cast<Eigen::Index>(selected_indices.size()); ++col) {
        const Eigen::Index idx = selected_indices[col];
        const auto v = eigenvectors.col(idx);

        candidates.col(ncand++) = v.real();
        if (v.imag().norm() > 1e-14 * std::max(v.real().norm(), 1.0)) {
            candidates.col(ncand++) = v.imag();
        }
    }
    candidates.conservativeResize(n_i, ncand);

    if (ncand == 0) { Z_i_.resize(n_i, 0); return; }

    // Rank-revealing QR: keep only an orthonormal basis of range(candidates).
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(candidates);
    qr.setThreshold(1e-12);
    const Eigen::Index rank = qr.rank();

    Z_i_ = qr.householderQ() * Eigen::MatrixXd::Identity(n_i, rank);
}

} // namespace schwarz2lvl
