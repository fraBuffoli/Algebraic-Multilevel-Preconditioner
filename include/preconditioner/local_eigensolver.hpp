#ifndef LOCAL_EIGENSOLVER_HPP
#define LOCAL_EIGENSOLVER_HPP

#include "sparse_matrix.hpp"
#include "partition_of_unity.hpp"
#include <Eigen/Core>

namespace schwarz2lvl {

/**
 * @class LocalEigensolver
 * @brief Solves the local generalized eigenvalue problem for spectral coarse spaces.
 * 
 * This class solves A_ii * z = lambda * A_tilde_ii * z using Eigen's dense generalized
 * eigensolver, filtering the eigenvectors based on a user-defined threshold tau.
 */
class LocalEigensolver {
public:
    /**
     * @brief Constructor setting the filtering threshold.
     * @param tau The spectral cutoff threshold.
     * @param nev The maximum number of selected eigenvectors.
     */
    explicit LocalEigensolver(double tau, Eigen::Index nev = 300) : tau_(tau), nev_(nev) {}

    /**
     * @brief Computes and filters the local eigenvectors.
     * 
     * @param A_ii The original local sparse matrix.
     * @param A_tilde_ii The modified local sparse matrix obtained via lumping.
     * @param pou The partition of unity for the subdomain.
     */
    void computeEigenpairs(const MatrixType& A_ii,
                           const MatrixType& A_tilde_ii,
                           const PartitionOfUnity& pou);

    /**
     * @brief Gets the matrix of filtered eigenvectors (local base Z_i).
     * @return Const reference to the dense matrix containing chosen eigenvectors as columns.
     */
    const Eigen::MatrixXd& getLocalBase() const { return Z_i_; }

    /**
     * @brief Gets the number of selected eigenvectors for this subdomain.
     * @return Number of chosen modes.
     */
    Eigen::Index numSelected() const { return Z_i_.cols(); }

private:
    double          tau_; // Cutoff threshold
    Eigen::MatrixXd Z_i_; // Filtered local eigenvectors block matrix (size n_i x chosen_modes)
    Eigen::Index nev_; // Max number of selected eigenvectors    
};

} // namespace schwarz2lvl

#endif // LOCAL_EIGENSOLVER_HPP
