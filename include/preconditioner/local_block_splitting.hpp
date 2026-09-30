/**
 * @file local_block_splitting.hpp
 * @brief Local block splitting matrix Atilda_ii using lumping in the overlap (Definition 3.1).
 */
#ifndef SCHWARZ2LVL_PRECONDITIONER_LOCAL_BLOCK_SPLITTING_HPP
#define SCHWARZ2LVL_PRECONDITIONER_LOCAL_BLOCK_SPLITTING_HPP

#include "subdomain.hpp"
#include "types.hpp"

namespace schwarz2lvl::prec{

/**
 * @brief Builds Atilda_ii of Definition 3.1.
 *
 * With the local ordering [Omega_I, Omega_Gamma],
 * \f[ \tilde A_{ii} = \begin{pmatrix} A_I & A_{I\Gamma} \\ A_{\Gamma I} & A_\Gamma - S_i \end{pmatrix},
 *     \qquad S_i = \mathrm{diag}(s_i),\; s_i(j) = \sum_{k\in\Omega^c_i} |A(j,k)|. \f]
 * The cost is O(n_i): only the diagonal of the overlap block changes. The
 * sums s_i were computed while receiving the ghost rows (Subdomain), so
 * no communication is needed here.
 *
 * @param sd Overlapping subdomain.
 * @return A~_ii (column-major, compressed).
 */
core::SpMat buildLocalBlockSplitting(const part::Subdomain& sd);

} // namespace schwarz2lvl

#endif // SCHWARZ2LVL_PRECONDITIONER_LOCAL_BLOCK_SPLITTING_HPP
