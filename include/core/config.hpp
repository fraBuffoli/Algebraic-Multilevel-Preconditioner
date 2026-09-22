/**
 * @file config.hpp
 * @brief Project-wide type aliases and numerical constants.
 */

#ifndef SCHWARZ2LVL_CONFIG_HPP
#define SCHWARZ2LVL_CONFIG_HPP
 
#include <cstdint>
#include <Eigen/Sparse>
#include <Eigen/Core>
 
namespace schwarz2lvl {
 
// Scalar type used for all matrix/vector entries. The paper works over
// C or R; we specialize to real double precision (the common case for
// CFD pressure/coupled systems), which keeps Eigen's dense generalized
// eigensolver (QZ algorithm) simple to use.
using Scalar = double;
 
// Type used for global degree-of-freedom indices 
using GlobalIndex = int;
 
// Type used for local (per-subdomain) indices
using LocalIndex = int;
 
// Default relative residual tolerance for the outer GMRES solve
constexpr Scalar kDefaultGmresTolerance = 1e-8;
 
// Default GMRES restart length.
constexpr int kDefaultGmresRestart = 30;
 
// Default maximum number of GMRES iterations (outer + restarts).
constexpr int kDefaultGmresMaxIterations = 1000;
 
// Default spectral threshold tau used to select coarse-space eigenvectors
// (Eq. 3.2): eigenvectors with |lambda| >= 1/tau are kept. The paper
// reports tau = 0.6 as a good default.
constexpr Scalar kDefaultTau = 0.6;
 
// Relative tolerance (wrt the largest local eigenvalue) used to decide
// whether a generalized eigenvalue's denominator beta is "zero"
constexpr Scalar kInfiniteEigenvalueRelTol = 1e-10;

// Convenience alias for Eigen's SparseMatrix forced to RowMajor (CSR) layout.
using MatrixType = Eigen::SparseMatrix<double, Eigen::RowMajor>;

// Convenience alias for Eigen's Dense Vector.
using VectorType = Eigen::VectorXd;
 
}  // namespace schwarz2lvl
 
#endif  // SCHWARZ2LVL_CONFIG_HPP