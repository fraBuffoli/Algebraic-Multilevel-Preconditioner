#ifndef LOCAL_BLOCK_SPLITTING_HPP
#define LOCAL_BLOCK_SPLITTING_HPP

#include "global_matrix.hpp"
#include "local_matrix.hpp"
#include "subdomain_topology.hpp"
#include <vector>

namespace schwarz2lvl {

/**
 * @class LocalBlockSplitting
 * @brief Implements the algebraic lumping technique on the subdomain boundary block.
 * 
 * According to Definition 3.1 of the paper, this class modifies the diagonal 
 * entries of the boundary subblock by incorporating the missing external connections.
 */
class LocalBlockSplitting {
public:
    /**
     * @brief Default constructor.
     */
    LocalBlockSplitting() = default;

    /**
     * @brief Performs algebraic lumping on the local submatrix.
     * 
     * Modifies the diagonal of the boundary entries of A_ii based on the global matrix connections.
     * 
     * @param global_A The global sparse matrix A.
     * @param local_A_ii The local extracted sparse matrix A_ii (modified in-place).
     * @param topology The computed topology of this specific subdomain.
     * @param partition_map The global METIS partition map array.
     */
    void applyLumping(const SparseMatrixWrapper& global_A,
                      MatrixType& local_A_ii,
                      const SubdomainTopology& topology) const;

    /**
     * @brief lumping on the local submatrix, using only local information (no global matrix needed).
     *
     * @param local_A The local matrix containing the subdomain's data.
     * @param local_A_ii The local square block to be modified in-place.
     */
    void applyLumping(const LocalMatrix& local_A,
                      MatrixType& local_A_ii) const;
};

} // namespace schwarz2lvl

#endif // LOCAL_BLOCK_SPLITTING_HPP
