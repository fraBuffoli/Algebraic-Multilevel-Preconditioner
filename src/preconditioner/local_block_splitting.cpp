/**
 * @file local_block_splitting.cpp
 * @brief Implementation of buildLocalBlockSplitting().
 */
#include "local_block_splitting.hpp"

namespace schwarz2lvl::prec {

core::SpMat buildLocalBlockSplitting(const part::Subdomain& sd)
{
    core::SpMat At = sd.Aii();
    const core::Vec& s = sd.lumpingSums();
    // Atilda_Gamma = A_Gamma - diag(s_i). coeffRef inserts the diagonal entry if it is
    // structurally missing (e.g. zero diagonal blocks of coupled systems).
    for (core::LocalIndex j = 0; j < sd.nGhost(); ++j) {
        const core::LocalIndex l = sd.nOwned() + j;
        At.coeffRef(l, l) -= s[j];
    }
    At.makeCompressed();
    return At;
}

} // namespace schwarz2lvl::prec
