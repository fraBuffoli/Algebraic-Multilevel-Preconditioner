/**
 * @file types.hpp
 * @brief Fundamental scalar, index and matrix types shared by the whole library.
 *
 * Two index types are used throughout the code:
 *  - @ref schwarz2lvl::LocalIndex  indexes unknowns *inside one process*
 *    (owned + ghost unknowns of the overlapping subdomain). It is the storage
 *    index type of every Eigen sparse matrix.
 *  - @ref schwarz2lvl::GlobalIndex indexes unknowns of the *global* system
 *    (after the METIS renumbering). It is 64-bit so that very large problems
 *    (and OpenFOAM builds with 64-bit labels) are supported.
 */
#ifndef SCHWARZ2LVL_CORE_TYPES_HPP
#define SCHWARZ2LVL_CORE_TYPES_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cstdint>

namespace schwarz2lvl::core {

/// @brief Floating point type (the paper works in C; this code is restricted to R).
using Scalar = double;

/// @brief Index type for process-local quantities (Eigen storage index).
using LocalIndex = int;

/// @brief Index type for global row/column numbers.
using GlobalIndex = std::int64_t;

/// @brief Column-major sparse matrix (used for factorizations).
using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, LocalIndex>;

/// @brief Row-major sparse matrix (used for sparse matrix-vector products).
using SpMatRow = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, LocalIndex>;

/// @brief Dense column vector.
using Vec = Eigen::VectorXd;

/// @brief Dense column-major matrix.
using Mat = Eigen::MatrixXd;

/// @brief Triplet with local indices (input of Eigen::SparseMatrix::setFromTriplets).
using Triplet = Eigen::Triplet<Scalar, LocalIndex>;

/**
 * @brief Coefficient of a globally indexed sparse matrix.
 *
 * Used to hand distributed contributions of the coarse matrix to the coarse
 * solver: duplicated (row, col) pairs are *summed*.
 * 
 * Better than: using GlobalTriplet = Eigen::Triplet<Scalar, LocalIndex>;
 */
struct GlobalTriplet {
    GlobalIndex row; ///< Global row index (0-based).
    GlobalIndex col; ///< Global column index (0-based).
    Scalar      val; ///< Value.
};

} // namespace schwarz2lvl::core

#endif // SCHWARZ2LVL_CORE_TYPES_HPP
