/**
 * @file two_level_schwarz.cpp
 * @brief Implementation of TwoLevelSchwarz.
 */
#include "two_level_schwarz.hpp"

#include "local_block_splitting.hpp"
#include "mpi_utils.hpp"
#include "timer.hpp"

#include <iomanip>
#include <ostream>
#include <stdexcept>

namespace schwarz2lvl::prec {

TwoLevelSchwarz::TwoLevelSchwarz(const part::Subdomain& sd, const linalg::DistributedMatrix& A, const core::SolverConfig& cfg)
    : sd_(sd), A_(A), correction_(cfg.coarse == "additive" ? Correction::Additive : Correction::Deflated)
{
    utils::ScopedTimer timer("setup");
    MPI_Comm comm = sd_.comm();
    const auto variant = (cfg.one_level == "asm") ? OneLevelSchwarz::Variant::ASM : OneLevelSchwarz::Variant::RAS;
    one_ = std::make_unique<OneLevelSchwarz>(sd_, variant, cfg.local_solver);

    // Local block splitting and local eigenproblem (no communication).
    std::string error;
    try {
        core::SpMat At;
        {
            utils::ScopedTimer t("setup.block_splitting");
            At = buildLocalBlockSplitting(sd_);
        }
        LocalEigensolverOptions eo;
        eo.tau = cfg.tau;
        eo.nev = cfg.nev;
        eo.tol = cfg.eig_tol;
        eo.maxit = cfg.eig_maxit;
        eo.ncv = cfg.eig_ncv;
        eo.shift_rel = cfg.shift_rel;
        eo.kernel_tol = cfg.kernel_tol;
        eo.kernel_probe = cfg.kernel_probe;
        eo.dense_threshold = cfg.dense_threshold;
        eo.local_solver = cfg.local_solver;
        eo.symmetric = sd_.system().symmetric;
        eig_ = LocalEigensolver(eo).compute(sd_.Aii(), At, sd_.pou());
    } catch (const std::exception& e) {
        error = e.what();
    }
    if (core::allreduceMin(error.empty() ? 1 : 0, comm) == 0)
        throw std::runtime_error("local eigensolver failed on at least one rank" +
                                 (error.empty() ? std::string() : (": " + error)));

    // ---- Coarse space and coarse operator.
    coarse_ = std::make_unique<CoarseSpace>(sd_, eig_.Z, cfg.orth_tol);
    if (coarse_->dimension() > 0)
        coarse_->assemble(cfg.coarse_solver, cfg.coarse_procs, cfg.local_solver, sd_.system().symmetric);

    // ---- Statistics.
    const int m = coarse_->localDimension();
    stats_.n0 = coarse_->dimension();
    stats_.m_min = core::allreduceMin(m, comm);
    stats_.m_max = core::allreduceMax(m, comm);
    stats_.m_avg = static_cast<double>(stats_.n0) / core::mpiSize(comm);
    stats_.kernel_subdomains = core::allreduceSum(eig_.kernel_dim > 0 ? 1 : 0, comm);
    stats_.kernel_dim_max = core::allreduceMax(eig_.kernel_dim, comm);
    stats_.kernel_part_total = core::allreduceSum(static_cast<core::GlobalIndex>(eig_.kernel_part), comm);
    stats_.dense_subdomains = core::allreduceSum(eig_.dense ? 1 : 0, comm);
    stats_.not_converged = core::allreduceSum(eig_.converged ? 0 : 1, comm);
    stats_.matvecs_max = static_cast<long>(core::allreduceMax(static_cast<core::GlobalIndex>(eig_.matvecs), comm));
}

std::string TwoLevelSchwarz::name() const
{
    return one_->name() + (correction_ == Correction::Deflated ? ",deflated" : ",additive");
}

void TwoLevelSchwarz::apply(const core::Vec& r, core::Vec& z) const
{
    if (coarse_->dimension() == 0) { // empty coarse space: one-level preconditioner
        one_->apply(r, z);
        return;
    }
    if (correction_ == Correction::Additive) {
        // z = R_0^T A_00^{-1} R_0 r + M_*^{-1} r
        coarse_->apply(r, x0_);
        one_->apply(r, z);
        z += x0_;
    } else {
        // z = x0 + M_*^{-1} (r - A x0),  x0 = R_0^T A_00^{-1} R_0 r
        coarse_->apply(r, x0_);
        A_.apply(x0_, w_);
        w_ = r - w_;
        one_->apply(w_, z1_);
        z = x0_ + z1_;
    }
}

void TwoLevelSchwarz::printStats(std::ostream& os) const
{
    if (core::mpiRank(sd_.comm()) != 0) return;
    os << "Coarse space\n"
       << "  n0                         " << stats_.n0 << "\n"
       << "  m_i min/avg/max            " << stats_.m_min << " / " << std::fixed << std::setprecision(1)
       << stats_.m_avg << " / " << stats_.m_max << "\n"
       << std::defaultfloat
       << "  subdomains with ker(A~)    " << stats_.kernel_subdomains << " (max dim " << stats_.kernel_dim_max << ")\n"
       << "  sum dim (L∩K)^⊥_K          " << stats_.kernel_part_total << "\n"
       << "  dense eigensolves          " << stats_.dense_subdomains << "\n"
       << "  Arnoldi not converged      " << stats_.not_converged << "\n"
       << "  max operator applications  " << stats_.matvecs_max << "\n";
    if (coarse_->dimension() > 0) os << "  coarse solver              " << coarse_->solver().name() << "\n";
}

} // namespace schwarz2lvl::prec
