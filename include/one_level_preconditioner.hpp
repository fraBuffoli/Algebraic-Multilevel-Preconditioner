#ifndef ONE_LEVEL_PRECONDITIONER_HPP
#define ONE_LEVEL_PRECONDITIONER_HPP

#include "preconditioner.hpp"
#include "subdomain_topology.hpp"
#include "restriction_operator.hpp"
#include "partition_of_unity.hpp"
#include <Eigen/SparseLU>

namespace schwarz2lvl {

/**
 * @class OneLevelPreconditioner
 * @brief Implements the classical restricted/additive Schwarz one-level preconditioner (ASM/RAS).
 * 
 * This class extracts the local subdomain matrix A_ii, factorizes it using a direct 
 * Sparse LU solver during setup, and applies local corrections during the Krylov solve phase.
 */
class OneLevelPreconditioner : public Preconditioner {
public:
    /**
     * @brief Default constructor creating an uninitialized preconditioner.
     */
    OneLevelPreconditioner() = default;

    /**
     * @brief Destructor. Automatically cleans up factorized matrices.
     */
    virtual ~OneLevelPreconditioner() = default;

    /**
     * @brief Initializes the preconditioner by extracting and factorizing the local subdomain matrix A_ii.
     * 
     * @param A The global sparse matrix.
     * @param topology The computed topology for this specific subdomain.
     * @param restriction The restriction operator R_i for this specific subdomain.
     * @param pou The partition of unity weights D_i for this specific subdomain.
     */
    void setup(const SparseMatrixWrapper& A,
               const SubdomainTopology& topology,
               const RestrictionOperator& restriction,
               const PartitionOfUnity& pou);

    /**
     * @brief Applies the 1-level Schwarz preconditioner: z = M_ASM^-1 * r.
     * 
     * Overrides the pure virtual method from the Preconditioner interface.
     * 
     * @param r Input global residual vector.
     * @param z Output global correction vector (accumulated).
     */
    virtual void apply(const VectorType& r, VectorType& z) const override;

    /**
     * @brief Gets read access to the local extracted sparse matrix A_ii.
     * @return Const reference to the local Eigen sparse matrix.
     */
    const MatrixType& getLocalMatrix() const { return A_ii_; }

private:
    MatrixType A_ii_; // Local extracted subdomain matrix (size n_i x n_i)
    RestrictionOperator restriction_; // Local restriction/prolongation operator R_i
    PartitionOfUnity pou_; // Local partition of unity weights D_i
    
    // Eigen's direct Sparse LU solver to handle the local linear system inversion efficiently
    mutable Eigen::SparseLU<MatrixType> solver_; 
};

} // namespace schwarz2lvl

#endif // ONE_LEVEL_PRECONDITIONER_HPP
