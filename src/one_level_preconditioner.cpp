#include "one_level_preconditioner.hpp"
#include <stdexcept>
#include <iostream>

namespace schwarz2lvl {

void OneLevelPreconditioner::setup(const SparseMatrixWrapper& A,
                                   const SubdomainTopology& topology,
                                   const RestrictionOperator& restriction,
                                   const PartitionOfUnity& pou) {
    
    // Copy operators locally to make this preconditioner fully self-contained
    restriction_ = restriction;
    pou_ = pou;

    const auto& global_matrix = A.getMatrix();
    const auto& local_indices = topology.getGlobalIndices();
    const Eigen::Index n_i = restriction_.localSize();

    // 1. Build a reverse mapping array to map global indices to local indices (0 to n_i-1).
    // This allows O(1) lookups when extracting submatrix coefficients.
    std::vector<int> global_to_local(A.rows(), -1);
    for (int j = 0; j < n_i; ++j) {
        global_to_local[local_indices[j]] = j;
    }

    // 2. Extract the local submatrix A_ii = R_i * A * R_i^T riga per riga.
    // Thanks to RowMajor layout, we can build the local matrix efficiently via triplets.
    std::vector<Eigen::Triplet<double>> local_triplets;
    Eigen::Index nnz_reserve = 0;
    for (int local_row = 0; local_row < n_i; ++local_row) {
        int global_row = local_indices[local_row];
        nnz_reserve += global_matrix.innerVector(global_row).nonZeros();
    }
    local_triplets.reserve(nnz_reserve); 

    for (int local_row = 0; local_row < n_i; ++local_row) {
        int global_row = local_indices[local_row];

        // Iterate over non-zero elements of the current global row using Eigen's InnerIterator
        for (MatrixType::InnerIterator it(global_matrix, global_row); it; ++it) {
            int global_col = static_cast<int>(it.col());
            int local_col = global_to_local[global_col];

            // If the column also belongs to our extended subdomain, save the coefficient
            if (local_col != -1) {
                local_triplets.push_back(Eigen::Triplet<double>(local_row, local_col, it.value()));
            }
        }
    }

    // Populate and compress the local sparse matrix structures
    A_ii_.resize(n_i, n_i);
    A_ii_.setFromTriplets(local_triplets.begin(), local_triplets.end());
    A_ii_.makeCompressed();

    // 3. Pre-factorize the local matrix using direct Sparse LU decomposition.
    // This expensive step is done ONLY ONCE during setup.
    solver_.analyzePattern(A_ii_);
    solver_.factorize(A_ii_);

    if (solver_.info() != Eigen::Success) {
        throw std::runtime_error("OneLevelPreconditioner Error: Direct factorization of local submatrix A_ii failed.");
    }
}

void OneLevelPreconditioner::apply(const VectorType& r, VectorType& z) const {
    
    VectorType r_local;
    VectorType y_local;

    // Step A: Restriction phase -> r_local = R_i * r
    restriction_.apply(r, r_local);

    // Step B: Local Solve phase -> y_local = A_ii^-1 * r_local
    // This performs a highly optimized forward/backward substitution using the pre-factorized LU arrays
    y_local = solver_.solve(r_local);
    if (solver_.info() != Eigen::Success) {
        throw std::runtime_error("OneLevelPreconditioner Solve Error: Local substitution failed.");
    }

    // Step C: Weighting phase -> y_local = D_i * y_local
    // We scale the local solution using the partition of unity diagonal entries
    const auto& weights = pou_.getWeights();
    y_local = y_local.cwiseProduct(weights);

    // Step D: Prolongation phase -> z = z + R_i^T * y_local
    // Accumulates the scaled local correction into the global solution vector
    restriction_.applyTranspose(y_local, z);
}

} // namespace schwarz2lvl
