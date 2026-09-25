/**
 * @file domain_decomposer.hpp
 * @brief OpenFOAM-like decomposition: rank 0 loads, partitions and scatters the system.
 */
#ifndef SCHWARZ2LVL_PARTITION_DOMAIN_DECOMPOSER_HPP
#define SCHWARZ2LVL_PARTITION_DOMAIN_DECOMPOSER_HPP

#include "config.hpp"
#include "csr_matrix.hpp"
#include "local_system.hpp"
#include "types.hpp"

#include <mpi.h>

#include <string>
#include <vector>

namespace schwarz2lvl::part {

/**
 * @brief Data that only rank 0 keeps after the decomposition.
 *
 * The permutation is needed to return the solution in the original ordering
 * (what `reconstructPar` does in OpenFOAM). Optionally, the permuted global
 * system is kept for testing purposes.
 */
struct DecompositionInfo {
    std::vector<core::GlobalIndex> new_to_old;      ///< Rank 0: original index of every renumbered unknown.
    std::string description;                        ///< Problem description.
    core::GlobalIndex nnz = 0;                      ///< Global number of nonzeros.
    std::vector<int> part;                          ///< Rank 0: METIS partition vector (original ordering).
    bool keep_global = false;                       ///< Input: keep the permuted global system on rank 0.
    CsrMatrix A_perm;                               ///< Rank 0 (if keep_global): permuted global matrix.
    std::vector<core::Scalar> b_perm;               ///< Rank 0 (if keep_global): permuted right-hand side.
};

/**
 * @class DomainDecomposer
 * @brief Simulates the OpenFOAM `decomposePar` + parallel assembly workflow.
 *
 * Steps (all global work is done on rank 0; this phase is
 * deliberately not optimized because in OpenFOAM it does not exist: every
 * processor assembles its own rows):
 *  1. read the Matrix Market file or run a built-in generator;
 *  2. partition the undirected graph G(A + A^T) with METIS into one subdomain
 *     per process;
 *  3. renumber the unknowns so that process p owns the contiguous range
 *     [ownership[p], ownership[p+1]);
 *  4. send to every process *only its owned rows* (CSR with global column
 *     indices) and the owned entries of the right-hand side.
 *
 * After decompose() no process holds any global data (except the optional
 * test copy on rank 0).
 */
class DomainDecomposer {
public:
    /**
     * @brief Loads, partitions and distributes the linear system (collective).
     * @param cfg  Configuration (input matrix / generator, rhs, METIS options).
     * @param comm Communicator (one subdomain per process).
     * @param info Rank-0 information (permutation, description); may not be null.
     * @return The local part of the system on every process.
     */
    static core::LocalSystem decompose(const core::SolverConfig& cfg, MPI_Comm comm, DecompositionInfo& info);

    /**
     * @brief Gathers a distributed vector on rank 0 in the *original* ordering (collective).
     * @param sys     Local system (ownership).
     * @param x_owned Owned entries.
     * @param info    Decomposition info (permutation used on rank 0).
     * @return The global vector on rank 0, empty elsewhere.
     */
    static std::vector<core::Scalar> gatherOriginalOrder(const core::LocalSystem& sys, const core::Vec& x_owned,
                                                   const DecompositionInfo& info);

    /**
     * @brief Gathers a distributed vector on rank 0 in the *renumbered* ordering (collective).
     * @param sys     Local system.
     * @param x_owned Owned entries.
     * @return The global vector on rank 0, empty elsewhere.
     */
    static std::vector<core::Scalar> gather(const core::LocalSystem& sys, const core::Vec& x_owned);
};

} // namespace schwarz2lvl::part

#endif // SCHWARZ2LVL_PARTITION_DOMAIN_DECOMPOSER_HPP

