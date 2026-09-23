#include "distributed_one_level_preconditioner.hpp"
#include <mpi.h>
#include <iostream>
#include <cstdlib>

namespace schwarz2lvl {

void DistributedOneLevelPreconditioner::setup(const LocalMatrix& local_A, const HaloExchange& halo,
                                              const VectorType& pou_weights) {
    halo_ = &halo;
    num_interior_ = halo.numInterior();
    n_i_ = local_A.rows();
    pou_weights_ = pou_weights;

    const MatrixType A_ii = local_A.extractLocalSquareBlock();
    solver_.analyzePattern(A_ii);
    solver_.factorize(A_ii);
    if (solver_.info() != Eigen::Success) {
        std::cerr << "DistributedOneLevelPreconditioner: fattorizzazione locale fallita." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    r_ext_.resize(n_i_);
    y_ext_.resize(n_i_);
}

void DistributedOneLevelPreconditioner::apply(const VectorType& r_owned, VectorType& z_owned) const {
    // 1) Extend the local residual vector to include halo values and perform halo exchange.
    r_ext_.head(num_interior_) = r_owned;
    halo_->exchange(r_ext_);

    // 2) Solve the local system A_ii * y_ext_ = r_ext_ and apply partition of unity weights.
    y_ext_ = solver_.solve(r_ext_);
    if (solver_.info() != Eigen::Success) {
        std::cerr << "DistributedOneLevelPreconditioner: sostituzione locale fallita." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    y_ext_ = y_ext_.cwiseProduct(pou_weights_);

    // 3) Scatter the local correction vector back to the owned portion of the global correction vector.
    halo_->scatterAddToOwner(y_ext_, z_owned);
}

} // namespace schwarz2lvl