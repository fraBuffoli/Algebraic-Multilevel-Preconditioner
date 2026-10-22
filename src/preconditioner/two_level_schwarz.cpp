/**
 * @file two_level_schwarz.cpp
 * @brief Implementation of TwoLevelSchwarz.
 */
#include "two_level_schwarz.hpp"

#include "local_block_splitting.hpp"
#include "logger.hpp"
#include "mpi_utils.hpp"
#include "timer.hpp"

#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace schwarz2lvl::prec {

TwoLevelSchwarz::TwoLevelSchwarz(const part::Subdomain& sd, const linalg::DistributedMatrix& A, const core::SolverConfig& cfg)
    : sd_(sd), A_(A), cfg_(cfg), correction_(cfg.coarse == "additive" ? Correction::Additive : Correction::Deflated)
{
    utils::ScopedTimer timer("setup");
    const auto variant = (cfg.one_level == "asm") ? OneLevelSchwarz::Variant::ASM : OneLevelSchwarz::Variant::RAS;
    one_ = std::make_unique<OneLevelSchwarz>(sd_, variant, localSolverSpec(cfg, sd_.system().symmetric));
    buildCoarse();
    version_ = sd_.valuesVersion();
}

UpdateLevel TwoLevelSchwarz::update(UpdateLevel requested)
{
    requested = agree(requested, sd_.comm());
    if (requested == UpdateLevel::Reuse || (requested == UpdateLevel::Refactorize && upToDate()))
        return UpdateLevel::Reuse;
    if (requested == UpdateLevel::Refactorize) {
        utils::ScopedTimer timer("update");
        one_->update(UpdateLevel::Refactorize);                  // numerical LU of A_ii
        if (coarse_->dimension() > 0) coarse_->refresh();        // A_00 = V^T A V, numerical LU
        version_ = sd_.valuesVersion();
        return UpdateLevel::Refactorize;
    }
    utils::ScopedTimer timer("setup");
    one_->update(UpdateLevel::Rebuild);                          // full LU of A_ii
    buildCoarse();                                               // eigenproblems, new V, new A_00
    version_ = sd_.valuesVersion();
    return UpdateLevel::Rebuild;
}

void TwoLevelSchwarz::buildCoarse()
{
    const core::SolverConfig& cfg = cfg_;
    MPI_Comm comm = sd_.comm();
    if (coarse_) coarse_fallbacks_ += coarse_->fallbacks();
    coarse_.reset(); // release the old coarse factorization first (Rebuild): lower memory peak

    // Local block splitting and local eigenproblem (no communication).
    std::string error;
    const double t_eig0 = MPI_Wtime();
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
        eo.nev_start = cfg.nev_start;
        eo.shift_rel = cfg.shift_rel;
        eo.kernel_tol = cfg.kernel_tol;
        eo.kernel_probe = cfg.kernel_probe;
        eo.dense_threshold = cfg.dense_threshold;
        eo.local_solver = cfg.local_solver;
        eo.factorization = cfg.local_factorization; // A~_ii + sigma I: Cholesky chain if A is symmetric (double)
        eo.symmetric = sd_.system().symmetric;
        eig_ = LocalEigensolver(eo).compute(sd_.Aii(), At, sd_.pou());
    } catch (const std::exception& e) {
        error = e.what();
    }
    if (utils::Log::verbosity() >= 2) { // one line per rank as soon as it is done: shows who is slow or stuck
        std::ostringstream s;
        s << "[rank " << core::mpiRank(comm) << "] eigensolver " << (error.empty() ? "done" : "FAILED") << " in "
          << std::fixed << std::setprecision(2) << MPI_Wtime() - t_eig0 << " s: n_i " << sd_.nLocal() << ", n_Gamma "
          << sd_.nGhost() << ", ker " << eig_.kernel_dim << ", selected " << eig_.Z.cols() << " of " << eig_.nev_used
          << " computed, operator applications "
          << eig_.matvecs << ", attempts " << eig_.attempts << (eig_.converged ? "" : ", NOT converged")
          << (error.empty() ? "" : ", " + error) << "\n";
        std::cerr << s.str() << std::flush;
    }
    if (core::allreduceMin(error.empty() ? 1 : 0, comm) == 0)
        throw std::runtime_error("local eigensolver failed on at least one rank" +
                                 (error.empty() ? std::string() : (": " + error)));

    // ---- Coarse space and coarse operator.
    coarse_ = std::make_unique<CoarseSpace>(sd_, eig_.Z, cfg.orth_tol);
    if (coarse_->dimension() > 0)
        coarse_->assemble(cfg.coarse_solver, cfg.coarse_procs, cfg.local_solver, cfg.coarse_factorization, sd_.system().symmetric,
                          static_cast<core::GlobalIndex>(cfg.coarse_root_below));

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
    stats_.retried = core::allreduceSum(eig_.attempts > 1 ? 1 : 0, comm);
    stats_.nev_used_max = core::allreduceMax(eig_.nev_used, comm);
    stats_.nev_used_avg = static_cast<double>(core::allreduceSum(eig_.nev_used, comm)) / core::mpiSize(comm);
    stats_.matvecs_max = static_cast<long>(core::allreduceMax(static_cast<core::GlobalIndex>(eig_.matvecs), comm));
    // Perturbed pivots (-1 = not reported): with PARDISO they switch iterative refinement back on.
    stats_.perturbed_aii_max = static_cast<long>(
        core::allreduceMax(static_cast<core::GlobalIndex>(one_->localSolver().perturbedPivots()), comm));
    stats_.perturbed_atilde_max =
        static_cast<long>(core::allreduceMax(static_cast<core::GlobalIndex>(eig_.perturbed_pivots), comm));
    stats_.local_factorization = one_->localSolver().name();
    stats_.atilde_factorization = eig_.dense ? std::string("dense") : eig_.backend;
    stats_.local_chain_fallbacks = core::allreduceSum(one_->localSolver().chainLevel() > 0 ? 1 : 0, comm);
    stats_.atilde_chain_fallbacks = core::allreduceSum(eig_.chain_level > 0 ? 1 : 0, comm);
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
    // Timers: the three parts of one application (coarse correction, extra SpMV of the
    // deflated variant, one-level solve), each split further inside CoarseSpace::apply()
    // and OneLevelSchwarz::apply(). Their sum is "solve.preconditioner" (Gmres).
    {
        utils::ScopedTimer t("solve.preconditioner.coarse");
        coarse_->apply(r, x0_); // x0 = R_0^T A_00^{-1} R_0 r
    }
    if (correction_ == Correction::Additive) {
        // z = R_0^T A_00^{-1} R_0 r + M_*^{-1} r
        utils::ScopedTimer t("solve.preconditioner.local");
        one_->apply(r, z);
        z += x0_;
    } else {
        // z = x0 + M_*^{-1} (r - A x0)
        {
            utils::ScopedTimer t("solve.preconditioner.spmv");
            A_.apply(x0_, w_);
        }
        w_ = r - w_;
        utils::ScopedTimer t("solve.preconditioner.local");
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
       << "  Arnoldi retried            " << stats_.retried << "\n"
       << "  Arnoldi nev avg/max        " << std::fixed << std::setprecision(1) << stats_.nev_used_avg << " / "
       << stats_.nev_used_max << std::defaultfloat << "\n"
       << "  max operator applications  " << stats_.matvecs_max << "\n";
    if (stats_.perturbed_aii_max >= 0 || stats_.perturbed_atilde_max >= 0)
        os << "  perturbed pivots A_ii / A~ " << stats_.perturbed_aii_max << " / " << stats_.perturbed_atilde_max
           << " (max over ranks)\n";
    os << "  A_ii factorization         " << stats_.local_factorization << " (rank 0; chain fallbacks on "
       << stats_.local_chain_fallbacks << " ranks)\n"
       << "  A~_ii factorization        " << stats_.atilde_factorization << " (rank 0; chain fallbacks on "
       << stats_.atilde_chain_fallbacks << " ranks)\n";
    if (coarse_->dimension() > 0) os << "  coarse solver              " << coarse_->solver().name() << "\n";
}

} // namespace schwarz2lvl::prec
