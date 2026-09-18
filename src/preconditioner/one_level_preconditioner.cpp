#include "one_level_preconditioner.hpp"
#include <stdexcept>
#include <iostream>
#include <mpi.h>

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
        std::cerr << "OneLevelPreconditioner Solve Error: Direct factorization of local submatrix A_ii failed." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // 4. Size the scratch buffers once, so apply() allocates nothing per iteration.
    r_local_.resize(n_i);
    y_local_.resize(n_i);
    z_this_.resize(A.rows());
    z_sum_.resize(A.rows());
}

void OneLevelPreconditioner::apply(const VectorType& r, VectorType& z) const {

    // Step A: Restriction -> r_local = R_i * r
    restriction_.apply(r, r_local_);

    // Step B: Local solve -> y_local = A_ii^-1 * r_local
    y_local_ = solver_.solve(r_local_);
    if (solver_.info() != Eigen::Success) {
        std::cerr << "OneLevelPreconditioner Solve Error: local substitution failed." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);   
    }

    // Step C: Partition of unity scaling -> y_local = D_i * y_local
    y_local_ = y_local_.cwiseProduct(pou_.getWeights());

    // Step D: Prolongation of this subdomain's term -> z_this = R_i^T * D_i * y_local
    z_this_.setZero();
    restriction_.applyTranspose(y_local_, z_this_);

    // Step E: M_RAS^-1 is the SUM over ALL subdomains (Eq. 2.3)
    MPI_Allreduce(z_this_.data(), z_sum_.data(),
                  static_cast<int>(z_this_.size()),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    z += z_sum_;
}
} // namespace schwarz2lvl
