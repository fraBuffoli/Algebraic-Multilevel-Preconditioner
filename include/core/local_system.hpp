/**
 * @file local_system.hpp
 * @brief Per-process piece of the distributed linear system A x = b.
 *
 * This structure is the **only interface** between the "decomposition" phase
 * (which, in this project, simulates what OpenFOAM does: rank 0 reads the
 * matrix, METIS partitions it and each rank receives its own rows) and the
 * preconditioner. In an OpenFOAM port, a LocalSystem is filled directly from
 * the processor-local `lduMatrix` (diagonal, upper, lower, interface/processor
 * patch coefficients) plus a `globalIndex` object: nothing global is needed.
 */
#ifndef SCHWARZ2LVL_CORE_LOCAL_SYSTEM_HPP
#define SCHWARZ2LVL_CORE_LOCAL_SYSTEM_HPP

#include "types.hpp"

#include <mpi.h>

#include <cstdint>
#include <vector>

namespace schwarz2lvl::core {

/**
 * @brief Rows of the global matrix owned by the calling process.
 *
 * Global unknowns are numbered so that every rank owns a *contiguous* range
 * [ownership[rank], ownership[rank+1]) (OpenFOAM `globalIndex` convention).
 * The owned rows are stored in CSR format with **global** column indices:
 * a column outside the owned range is a coupling to a neighbouring process
 * (what OpenFOAM stores in processor-patch interface coefficients).
 */
struct LocalSystem {
    MPI_Comm comm = MPI_COMM_WORLD;         ///< Communicator of the decomposition.
    GlobalIndex n_global = 0;               ///< Global number of unknowns n.
    std::vector<GlobalIndex> ownership;     ///< Ownership offsets, size nranks+1.
    std::vector<GlobalIndex> row_ptr;       ///< CSR row pointer of owned rows (size n_owned+1).
    std::vector<GlobalIndex> col;           ///< CSR global column indices.
    std::vector<Scalar> val;                ///< CSR values.
    std::vector<Scalar> rhs;                ///< Owned entries of the right-hand side b.
    int block_size = 1;                     ///< Unknowns per cell (1 = scalar problem).
    bool symmetric = false;                 ///< True if A is known to be symmetric.

    /// @brief First owned global row.
    GlobalIndex rowBegin() const;
    /// @brief One past the last owned global row.
    GlobalIndex rowEnd() const;
    /// @brief Number of owned rows.
    LocalIndex nOwned() const { return static_cast<LocalIndex>(row_ptr.empty() ? 0 : row_ptr.size() - 1); }
};

inline GlobalIndex LocalSystem::rowBegin() const
{
    int r = 0;
    MPI_Comm_rank(comm, &r);
    return ownership[static_cast<std::size_t>(r)];
}

inline GlobalIndex LocalSystem::rowEnd() const
{
    int r = 0;
    MPI_Comm_rank(comm, &r);
    return ownership[static_cast<std::size_t>(r) + 1];
}

} // namespace schwarz2lvl::core

#endif // SCHWARZ2LVL_CORE_LOCAL_SYSTEM_HPP
