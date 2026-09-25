/**
 * @file mpi_utils.hpp
 * @brief MPI helpers: datatype traits, error checking, ownership lookup.
 */
#ifndef SCHWARZ2LVL_CORE_MPI_UTILS_HPP
#define SCHWARZ2LVL_CORE_MPI_UTILS_HPP

#include "types.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace schwarz2lvl::core {

/**
 * @brief Maps a C++ type to the corresponding MPI datatype.
 * @tparam T C++ type (specialized for the types used in the library).
 */
template <typename T> MPI_Datatype mpiType();
/// @cond
template <> inline MPI_Datatype mpiType<double>() { return MPI_DOUBLE; }
template <> inline MPI_Datatype mpiType<int>() { return MPI_INT; }
template <> inline MPI_Datatype mpiType<GlobalIndex>() { return MPI_INT64_T; }
template <> inline MPI_Datatype mpiType<unsigned long>() { return MPI_UNSIGNED_LONG; }
template <> inline MPI_Datatype mpiType<char>() { return MPI_CHAR; }
/// @endcond

/**
 * @brief Throws a std::runtime_error if an MPI call did not return MPI_SUCCESS.
 * @param err  Return code of the MPI call.
 * @param what Human readable description of the call.
 */
inline void mpiCheck(int err, const char* what)
{
    if (err != MPI_SUCCESS) {
        throw std::runtime_error(std::string("MPI error in ") + what);
    }
}

/// @brief Rank of the calling process in @p comm.
inline int mpiRank(MPI_Comm comm)
{
    int r = 0;
    MPI_Comm_rank(comm, &r);
    return r;
}

/// @brief Number of processes in @p comm.
inline int mpiSize(MPI_Comm comm)
{
    int s = 1;
    MPI_Comm_size(comm, &s);
    return s;
}

/**
 * @brief Returns the rank owning global index @p g.
 *
 * Ownership is contiguous (OpenFOAM `globalIndex` style): rank p owns
 * [ownership[p], ownership[p+1]).
 *
 * @param ownership Offsets array of size nranks+1.
 * @param g         Global index.
 * @return Owner rank.
 */
inline int ownerOf(const std::vector<GlobalIndex>& ownership, GlobalIndex g)
{
    auto it = std::upper_bound(ownership.begin(), ownership.end(), g);
    return static_cast<int>(std::distance(ownership.begin(), it)) - 1;
}

/**
 * @brief Sum-reduction of a scalar over @p comm.
 * @tparam T Arithmetic type with an MPI datatype.
 */
template <typename T> T allreduceSum(T v, MPI_Comm comm)
{
    T out{};
    MPI_Allreduce(&v, &out, 1, mpiType<T>(), MPI_SUM, comm);
    return out;
}

/// @brief Max-reduction of a scalar over @p comm.
template <typename T> T allreduceMax(T v, MPI_Comm comm)
{
    T out{};
    MPI_Allreduce(&v, &out, 1, mpiType<T>(), MPI_MAX, comm);
    return out;
}

/// @brief Min-reduction of a scalar over @p comm.
template <typename T> T allreduceMin(T v, MPI_Comm comm)
{
    T out{};
    MPI_Allreduce(&v, &out, 1, mpiType<T>(), MPI_MIN, comm);
    return out;
}

/**
 * @brief Converts a vector of counts to an exclusive prefix-sum displacement vector.
 * @param counts Per-rank counts.
 * @return Displacements (same size as @p counts).
 */
inline std::vector<int> displacements(const std::vector<int>& counts)
{
    std::vector<int> d(counts.size(), 0);
    for (std::size_t i = 1; i < counts.size(); ++i) d[i] = d[i - 1] + counts[i - 1];
    return d;
}

} // namespace schwarz2lvl::core

#endif // SCHWARZ2LVL_CORE_MPI_UTILS_HPP
