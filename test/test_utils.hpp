/**
 * @file test_utils.hpp
 * @brief Helpers shared by the tests: global reference data and checks.
 *
 * The tests compare the distributed implementation with a sequential
 * reference computed from the *global* permuted matrix, which the tests (and
 * only the tests) broadcast to every process. Use small problems.
 */
#ifndef SCHWARZ2LVL_TEST_UTILS_HPP
#define SCHWARZ2LVL_TEST_UTILS_HPP

#include "csr_matrix.hpp"
#include "mpi_utils.hpp"
#include "types.hpp"

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace schwarz2lvl::test {

/// @brief Global failure counter of the running test program.
inline int& failures()
{
    static int f = 0;
    return f;
}

/**
 * @brief Records a check (collective: the result is combined over all ranks).
 * @param ok   Local result.
 * @param what Description printed by rank 0.
 * @param comm Communicator.
 */
inline void check(bool ok, const std::string& what, ::MPI_Comm comm = MPI_COMM_WORLD)
{
    int loc = ok ? 1 : 0, glob = 0;
    MPI_Allreduce(&loc, &glob, 1, MPI_INT, MPI_MIN, comm);
    if (core::mpiRank(comm) == 0) std::cout << (glob ? "  [ OK ] " : "  [FAIL] ") << what << std::endl;
    if (!glob) ++failures();
}

/// @brief Broadcasts a CSR matrix from rank 0 to all ranks.
inline void bcastCsr(part::CsrMatrix& A, ::MPI_Comm comm)
{
    long long hdr[3] = {static_cast<long long>(A.nrows), static_cast<long long>(A.nnz()), A.symmetric ? 1 : 0};
    MPI_Bcast(hdr, 3, MPI_LONG_LONG, 0, comm);
    A.nrows = A.ncols = static_cast<core::GlobalIndex>(hdr[0]);
    A.symmetric = hdr[2] != 0;
    A.row_ptr.resize(static_cast<std::size_t>(hdr[0]) + 1);
    A.col.resize(static_cast<std::size_t>(hdr[1]));
    A.val.resize(static_cast<std::size_t>(hdr[1]));
    MPI_Bcast(A.row_ptr.data(), static_cast<int>(A.row_ptr.size()), core::mpiType<core::GlobalIndex>(), 0, comm);
    MPI_Bcast(A.col.data(), static_cast<int>(A.col.size()), core::mpiType<core::GlobalIndex>(), 0, comm);
    MPI_Bcast(A.val.data(), static_cast<int>(A.val.size()), MPI_DOUBLE, 0, comm);
}

/// @brief Converts a (small) CSR matrix to an Eigen sparse matrix.
inline core::SpMat toEigen(const part::CsrMatrix& A)
{
    std::vector<core::Triplet> t;
    t.reserve(A.val.size());
    for (core::GlobalIndex i = 0; i < A.nrows; ++i)
        for (std::int64_t k = A.row_ptr[static_cast<std::size_t>(i)]; k < A.row_ptr[static_cast<std::size_t>(i) + 1]; ++k)
            t.emplace_back(static_cast<core::LocalIndex>(i), static_cast<core::LocalIndex>(A.col[static_cast<std::size_t>(k)]),
                           A.val[static_cast<std::size_t>(k)]);
    core::SpMat M(static_cast<core::LocalIndex>(A.nrows), static_cast<core::LocalIndex>(A.ncols));
    M.setFromTriplets(t.begin(), t.end());
    return M;
}

/// @brief Formats a number in scientific notation (for check labels).
inline std::string sci(double v)
{
    std::ostringstream os;
    os.precision(2);
    os << std::scientific << v;
    return os.str();
}

/// @brief Relative difference ||a - b|| / max(||b||, tiny).
inline double relDiff(const core::Vec& a, const core::Vec& b)
{
    return (a - b).norm() / std::max(b.norm(), 1e-300);
}

} // namespace schwarz2lvl::test

#endif // SCHWARZ2LVL_TEST_UTILS_HPP
