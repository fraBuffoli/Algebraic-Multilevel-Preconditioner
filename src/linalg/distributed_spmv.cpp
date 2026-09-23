#include "distributed_spmv.hpp"

namespace schwarz2lvl {

void DistributedSpMV::setup(const LocalMatrix& local_A, const HaloExchange& halo) {
    halo_ = &halo;
    num_interior_ = halo.numInterior();
    n_i_ = local_A.rows();

    const MatrixType square = local_A.extractLocalSquareBlock();
    A_interior_local_ = square.topRows(num_interior_);   

    x_ext_.resize(n_i_);
}

void DistributedSpMV::apply(const VectorType& x_owned, VectorType& y_owned) const {
    x_ext_.head(num_interior_) = x_owned;
    halo_->exchange(x_ext_); 
    y_owned.noalias() = A_interior_local_ * x_ext_;
}

} // namespace schwarz2lvl