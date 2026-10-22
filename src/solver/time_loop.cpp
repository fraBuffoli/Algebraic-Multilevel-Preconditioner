/**
 * @file time_loop.cpp
 * @brief Implementation of runTimeLoop().
 */
#include "time_loop.hpp"

#include "distributed_matrix.hpp"
#include "domain_decomposer.hpp"
#include "gmres.hpp"
#include "logger.hpp"
#include "mpi_utils.hpp"
#include "one_level_schwarz.hpp"
#include "reuse_policy.hpp"
#include "subdomain.hpp"
#include "timer.hpp"
#include "two_level_schwarz.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <memory>
#include <ostream>
#include <random>

namespace schwarz2lvl::solv {

namespace {

/// @brief Random right-hand side of (step, corrector), original ordering, distributed.
core::Vec randomRhs(const core::SolverConfig& cfg, int step, int corr, const core::LocalSystem& sys,
                    const part::DecompositionInfo& info)
{
    std::vector<core::Scalar> bg;
    if (core::mpiRank(sys.comm) == 0) {
        std::seed_seq seq{cfg.seed, static_cast<unsigned>(step), static_cast<unsigned>(corr), 0x5eedu};
        std::mt19937_64 gen(seq);
        std::uniform_real_distribution<core::Scalar> dist(-1.0, 1.0);
        bg.resize(static_cast<std::size_t>(sys.n_global));
        for (auto& v : bg) v = dist(gen);
    }
    return part::DomainDecomposer::scatterOriginalOrder(sys, bg, info);
}

/// @brief Wall time between two barriers around @p f (identical on every rank up to the barrier skew).
template <typename F> double timed(MPI_Comm comm, F&& f)
{
    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();
    f();
    MPI_Barrier(comm);
    return MPI_Wtime() - t0;
}

} // namespace

std::vector<SolveRecord> runTimeLoop(const core::SolverConfig& cfg, MPI_Comm comm, std::ostream& out)
{
    const int np = core::mpiSize(comm);
    part::DecompositionInfo info;
    core::LocalSystem sys = part::DomainDecomposer::decompose(cfg, comm, info);
    out << "Problem\n  " << info.description << "\n  block size " << sys.block_size
        << (sys.symmetric ? ", symmetric" : ", nonsymmetric") << "\n";

    part::Subdomain sd(sys, cfg.pou);
    linalg::DistributedMatrix A(sd);
    // A one-level preconditioner has nothing that ages (no coarse basis): a Rebuild would only redo the
    // symbolic analysis, so "auto" behaves as "refactor" (otherwise noise in the iteration counts of
    // one-level methods triggers useless rebuilds).
    prec::ReusePolicy::Mode mode = prec::ReusePolicy::parseMode(cfg.reuse);
    const bool one_level = (cfg.coarse == "none");
    if (one_level && mode == prec::ReusePolicy::Mode::Auto) mode = prec::ReusePolicy::Mode::Refactor;
    prec::ReusePolicy policy(mode, cfg.rebuild_growth, cfg.rebuild_every);
    const Gmres gmres(cfg.restart, cfg.rtol, cfg.maxit);
    std::unique_ptr<prec::Preconditioner> M;
    const prec::TwoLevelSchwarz* two = nullptr;

    std::ofstream csv;
    if (core::mpiRank(comm) == 0 && !cfg.steps_csv.empty()) {
        std::ifstream probe(cfg.steps_csv);
        const bool header = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
        probe.close();
        csv.open(cfg.steps_csv, std::ios::app);
        if (header)
            csv << "problem,np,reuse,rebuild_growth,rebuild_every,step,corrector,level,t_update,t_solve,iterations,"
                   "per_decade,converged,true_rel_res,n0,wasted\n";
    }

    out << "Time loop: " << cfg.steps << " steps x " << cfg.correctors << " correctors, reuse " << cfg.reuse
        << (one_level && cfg.reuse == "auto" ? " (one-level: no coarse basis to rebuild, as refactor)" : "") << "\n"
        << "   step corr  level         update[s]  solve[s]  its  it/decade      n0  wasted[s]\n";
    std::vector<SolveRecord> recs;
    for (int step = 0; step < cfg.steps; ++step) {
        // New matrix values (outside the timers: in OpenFOAM the matrix is assembled by the solver).
        if (step > 0) part::DomainDecomposer::updateValues(cfg, step, sys, info);
        for (int corr = 0; corr < cfg.correctors; ++corr) {
            SolveRecord r;
            r.step = step;
            r.corrector = corr;
            const core::Vec b = (cfg.rhs == "random")
                                    ? randomRhs(cfg, step, corr, sys, info)
                                    : core::Vec(Eigen::Map<const core::Vec>(sys.rhs.data(), sys.nOwned()));
            r.t_update = timed(comm, [&] {
                if (!M) { // initial setup
                    if (cfg.coarse == "none") {
                        M = std::make_unique<prec::OneLevelSchwarz>(
                            sd, cfg.one_level == "asm" ? prec::OneLevelSchwarz::Variant::ASM : prec::OneLevelSchwarz::Variant::RAS,
                            cfg.local_solver);
                    } else {
                        auto p = std::make_unique<prec::TwoLevelSchwarz>(sd, A, cfg);
                        two = p.get();
                        M = std::move(p);
                    }
                    r.level = prec::UpdateLevel::Rebuild;
                } else {
                    sd.updateValues(); // the operator A follows the new values in any case
                    r.level = M->update(policy.decide(!M->upToDate()));
                }
            });
            core::Vec x = core::Vec::Zero(b.size());
            GmresResult res;
            r.t_solve = timed(comm, [&] { res = gmres.solve(A, *M, b, x); });
            const double r0 = res.history.empty() ? 1.0 : res.history.front();
            const double reduction = (r0 > 0.0) ? res.true_rel_residual / r0 : 0.0;
            policy.record(r.level, res.iterations, reduction, res.converged, r.t_update, r.t_solve);
            r.wasted = policy.wasted();
            r.iterations = res.iterations;
            r.per_decade = prec::ReusePolicy::perDecade(res.iterations, reduction);
            r.converged = res.converged;
            r.true_rel_residual = res.true_rel_residual;
            r.n0 = two ? two->stats().n0 : 0;
            recs.push_back(r);

            out << std::right << std::setw(7) << step << std::setw(5) << corr << "  " << std::left << std::setw(12)
                << prec::toString(r.level) << std::right << std::fixed << std::setprecision(4) << std::setw(10)
                << r.t_update << std::setw(10) << r.t_solve << std::setw(5) << r.iterations << std::setprecision(2)
                << std::setw(11) << r.per_decade << std::setw(8) << r.n0 << std::setprecision(4) << std::setw(11)
                << r.wasted << (r.converged ? "" : "  NOT converged")
                << std::defaultfloat << "\n";
            if (csv.is_open())
                csv << '"' << info.description << '"' << ',' << np << ',' << cfg.reuse << ',' << cfg.rebuild_growth << ','
                    << cfg.rebuild_every << ',' << step << ',' << corr << ',' << prec::toString(r.level) << ','
                    << r.t_update << ',' << r.t_solve << ',' << r.iterations << ',' << r.per_decade << ','
                    << (r.converged ? 1 : 0) << ',' << r.true_rel_residual << ',' << r.n0 << ',' << r.wasted << "\n";
        }
    }
    // Numerical refactorizations that had to fall back to a full factorization (e.g. PARDISO with
    // more perturbed pivots than the full factorization): local = summed over the ranks.
    {
        const long loc = M ? M->localFallbacks() : 0, crs = M ? M->coarseFallbacks() : 0;
        const long loc_sum = core::allreduceSum(static_cast<core::GlobalIndex>(loc), comm);
        const int loc_ranks = core::allreduceSum(loc > 0 ? 1 : 0, comm);
        const long crs_max = core::allreduceMax(static_cast<core::GlobalIndex>(crs), comm);
        out << "Refactorizations falling back to a full factorization: local " << loc_sum << " (on " << loc_ranks
            << " ranks), coarse " << crs_max << "\n";
    }
    if (two) two->printStats(out);
    return recs;
}

void printTimeLoopSummary(const std::vector<SolveRecord>& recs, const core::SolverConfig& cfg, MPI_Comm comm,
                          std::ostream& out)
{
    if (core::mpiRank(comm) != 0 || recs.empty()) return;
    int count[3] = {0, 0, 0}, itmax = 0, nconv = 0;
    double tlev[3] = {0.0, 0.0, 0.0}, tsolve = 0.0, its = 0.0;
    for (std::size_t k = 0; k < recs.size(); ++k) {
        const auto& r = recs[k];
        const int l = static_cast<int>(r.level);
        if (k > 0) { // the first solve is the initial setup, reported separately
            ++count[l];
            tlev[l] += r.t_update;
        }
        tsolve += r.t_solve;
        its += r.iterations;
        itmax = std::max(itmax, r.iterations);
        nconv += r.converged ? 1 : 0;
    }
    const double tinit = recs.front().t_update, tupd = tlev[0] + tlev[1] + tlev[2];
    out << "Time loop summary (reuse " << cfg.reuse << ", " << recs.size() << " solves, " << nconv << " converged)\n"
        << std::fixed << std::setprecision(3)
        << "  initial setup              " << tinit << " s\n"
        << "  updates                    " << tupd << " s   (reuse " << count[0] << " x, refactorize " << count[1]
        << " x " << (count[1] ? tlev[1] / count[1] : 0.0) << " s, rebuild " << count[2] << " x "
        << (count[2] ? tlev[2] / count[2] : 0.0) << " s)\n"
        << "  solves                     " << tsolve << " s   (iterations avg " << std::setprecision(1)
        << its / static_cast<double>(recs.size()) << ", max " << itmax << ")\n"
        << std::setprecision(3) << "  total setup + updates + solves " << tinit + tupd + tsolve << " s\n"
        << std::defaultfloat;
}

} // namespace schwarz2lvl::solv
