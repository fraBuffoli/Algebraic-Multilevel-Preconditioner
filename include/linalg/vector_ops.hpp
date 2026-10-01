/**
 * @file vector_ops.hpp
 * @brief Global reductions on distributed vectors (owned parts only).
 *
 * A distributed vector is stored as its *owned* entries on every process
 * (no overlap), so global inner products are local dot products followed by
 * one MPI_Allreduce, exactly as OpenFOAM's `gSumProd`.
 */
#ifndef SCHWARZ2LVL_LINALG_VECTOR_OPS_HPP
#define SCHWARZ2LVL_LINALG_VECTOR_OPS_HPP

#include "types.hpp"

#include <mpi.h>

#include <cmath>

namespace schwarz2lvl::linalg {

/**
 * @brief Global inner product x^T y (collective).
 * @param x    Owned entries of x.
 * @param y    Owned entries of y.
 * @param comm Communicator.
 */
inline core::Scalar dot(const core::Vec& x, const core::Vec& y, MPI_Comm comm)
{
    core::Scalar loc = x.dot(y), glob = 0.0;
    MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, comm);
    return glob;
}

/**
 * @brief Global Euclidean norm (collective).
 * @param x    Owned entries.
 * @param comm Communicator.
 */
inline core::Scalar norm2(const core::Vec& x, MPI_Comm comm) { return std::sqrt(dot(x, x, comm)); }

/**
 * @brief Several inner products with a single reduction: h = V(:, 0:k)^T w (collective).
 * @param V    Owned rows of the basis (column-major, at least k columns).
 * @param k    Number of columns used.
 * @param w    Owned entries of w.
 * @param comm Communicator.
 * @return Vector of the k inner products.
 */
inline core::Vec multiDot(const core::Mat& V, Eigen::Index k, const core::Vec& w, MPI_Comm comm)
{
    core::Vec loc = V.leftCols(k).transpose() * w;
    core::Vec glob(k);
    MPI_Allreduce(loc.data(), glob.data(), static_cast<int>(k), MPI_DOUBLE, MPI_SUM, comm);
    return glob;
}

} // namespace schwarz2lvl::linalg

#endif // SCHWARZ2LVL_LINALG_VECTOR_OPS_HPP
