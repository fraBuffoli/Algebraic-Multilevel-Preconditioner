#ifndef KRYLOV_BICGSTAB_HPP
#define KRYLOV_BICGSTAB_HPP

#include "krylov_solver.hpp"

namespace schwarz2lvl {

/**
 * @class KrylovBicgstab
 * @brief Implements the Preconditioned BiConjugate Gradient Stabilized (BiCGStab) solver.
 * 
 * This class inherits from KrylovSolver and implements a fully parallel, 
 * right-preconditioned BiCGStab solver utilizing native MPI operations.
 */
class KrylovBicgstab : public KrylovSolver {
public:
    KrylovBicgstab(int max_iter, double tolerance) : 
        KrylovSolver(max_iter, tolerance) {};

    virtual ~KrylovBicgstab() = default;

    virtual bool solve(const SparseMatrixWrapper& A,
                       const VectorType& b,
                       VectorType& x,
                       const Preconditioner& prec) const override;
};

} // namespace schwarz2lvl

#endif // KRYLOV_BICGSTAB_HPP
