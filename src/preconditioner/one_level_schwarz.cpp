/**
 * @file one_level_schwarz.cpp
 * @brief Implementation of OneLevelSchwarz.
 */
#include "one_level_schwarz.hpp"

#include "mpi_utils.hpp"
#include "timer.hpp"

#include <stdexcept>

namespace schwarz2lvl::prec {

OneLevelSchwarz::OneLevelSchwarz(const part::Subdomain& sd, Variant variant, const std::string& local_solver)
    : sd_(sd), variant_(variant), solver_(linalg::makeLocalSolver(local_solver))
{
    utils::ScopedTimer timer("setup.local_factorization");
    const bool ok = solver_->factorize(sd_.Aii());
    // Collective check: every rank throws, nobody hangs in a later collective.
    const int all_ok = core::allreduceMin(ok ? 1 : 0, sd_.comm());
    if (!all_ok)
        throw std::runtime_error("factorization of A_ii failed on at least one rank" +
                                 (ok ? std::string() : (": " + solver_->lastError())));
    rloc_ = core::Vec::Zero(sd_.nLocal());
}

void OneLevelSchwarz::apply(const core::Vec& r, core::Vec& z) const
{
    const core::LocalIndex nI = sd_.nOwned();
    rloc_.head(nI) = r;
    sd_.halo().forward(rloc_);          // R_i r
    solver_->solve(rloc_, yloc_);       // A_ii^{-1} R_i r
    if (variant_ == Variant::RAS) {
        yloc_.array() *= sd_.pou().array(); // D_i A_ii^{-1} R_i r
        if (!sd_.booleanPou()) sd_.halo().reverseAdd(yloc_);
    } else {
        sd_.halo().reverseAdd(yloc_);   // sum_j R_j^T A_jj^{-1} R_j r
    }
    z = yloc_.head(nI);
}

} // namespace schwarz2lvl::prec
