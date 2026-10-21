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

#include <cstdint>
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
    std::uint64_t pattern_hash = 0;                 ///< Rank 0: hash of the sparsity pattern (original ordering).
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
     * @brief Replaces the values of the distributed system with those of step @p step (collective).
     *
     * Rank 0 loads the matrix of the step (generator at pseudo-time @p step, or the file
     * obtained by formatting `--matrix` with @p step), checks that its sparsity pattern is the
     * one of the decomposition (hash) and sends every rank the *values* of its rows, in the
     * order of `sys.col`, plus its right-hand side. Partition, numbering and pattern are kept:
     * this is what happens in OpenFOAM, where each processor reassembles its matrix on the same
     * mesh at every time step.
     * @throws std::runtime_error if the pattern, the size or the symmetry changed.
     */
    static void updateValues(const core::SolverConfig& cfg, int step, core::LocalSystem& sys, DecompositionInfo& info);

    /**
     * @brief Distributes a global vector given on rank 0 in the *original* ordering (collective).
     * @param sys  Local system (ownership).
     * @param xg   Rank 0: global vector, original ordering (ignored elsewhere).
     * @param info Decomposition info (permutation on rank 0).
     * @return The owned entries.
     */
    static core::Vec scatterOriginalOrder(const core::LocalSystem& sys, const std::vector<core::Scalar>& xg,
                                          const DecompositionInfo& info);

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

