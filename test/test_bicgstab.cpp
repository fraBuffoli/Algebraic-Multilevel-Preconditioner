/**
 * @file test_bicgstab.cpp
 * @brief BiCGStab with the two-level Schwarz preconditioner (RAS + deflated, the nonsymmetric default).
 *
 *  1. convergence (true residual) on a symmetric (bubbles3d) and a nonsymmetric (convdiff3d) problem,
 *     exact and AMG local solves; iterations = applications of the preconditioner (counted), history
 *     of one entry per application; GMRES with the same preconditioner for reference;
 *  2. OpenFOAM stopping test: stops at the first application whose residual (s at the half step, r at
 *     the full step) is below relTol; normalisation and initial residual of the caller's x0;
 *  3. 2-norm test; unpreconditioned BiCGStab; budget of applications (maxit odd: stops at a half step
 *     with a consistent x);
 *  4. time loop with BiCGStab (refactorizations and rebuilds);
 *  5. configuration: --krylov gmres|bicgstab.
 */
#include "bicgstab.hpp"
#include "config.hpp"
#include "distributed_matrix.hpp"
#include "domain_decomposer.hpp"
#include "gmres.hpp"
#include "krylov.hpp"
#include "logger.hpp"
#include "subdomain.hpp"
#include "test_utils.hpp"
#include "time_loop.hpp"
#include "two_level_schwarz.hpp"
#include "vector_ops.hpp"

#include <cmath>
#include <iostream>
#include <memory>

using namespace schwarz2lvl;

namespace {

double norm1(const core::Vec& v, MPI_Comm comm)
{
    double s = v.cwiseAbs().sum();
    MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_DOUBLE, MPI_SUM, comm);
    return s;
}

core::Vec residual(const linalg::DistributedMatrix& A, const core::Vec& b, const core::Vec& x)
{
    core::Vec Ax;
    A.apply(x, Ax);
    return b - Ax;
}

/// @brief Counts the applications of a preconditioner.
class Counting final : public prec::Preconditioner {
public:
    explicit Counting(const prec::Preconditioner& M) : M_(M) {}
    void apply(const core::Vec& r, core::Vec& z) const override
    {
        ++count;
        M_.apply(r, z);
    }
    std::string name() const override { return M_.name(); }
    mutable int count = 0;

private:
    const prec::Preconditioner& M_;
};

core::SolverConfig bubbles()
{
    core::SolverConfig cfg;
    cfg.generate = "bubbles3d";
    cfg.nx = cfg.ny = cfg.nz = 24;
    cfg.nbubbles = 20;
    cfg.radius = 0.1;
    cfg.rhs = "generated";
    cfg.tau = 0.3;
    cfg.nev = 8;
    cfg.dense_threshold = 0;
    cfg.coarse_solver = "root";
    return cfg;
}

/// @brief Test 1 on one problem: BiCGStab vs GMRES, exact and AMG local solves.
void checkConvergence(const core::SolverConfig& base, const std::string& label)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    part::DecompositionInfo info;
    const core::LocalSystem sys = part::DomainDecomposer::decompose(base, comm, info);
    part::Subdomain sd(sys, base.pou);
    linalg::DistributedMatrix A(sd);
    const core::Vec b = (base.rhs == "generated") ? core::Vec(Eigen::Map<const core::Vec>(sys.rhs.data(), sys.nOwned()))
                                                   : core::Vec::Ones(sys.nOwned());
    for (const std::string approx : {"none", "amg"}) {
        core::SolverConfig c = base;
        c.local_approx = approx;
        const prec::TwoLevelSchwarz M(sd, A, c);
        Counting Mc(M);
        core::Vec x = core::Vec::Zero(sys.nOwned());
        const solv::KrylovResult r = solv::BiCgStab(1e-8, 1000).solve(A, Mc, b, x);
        const double t2 = linalg::norm2(residual(A, b, x), comm) / linalg::norm2(b, comm);
        core::Vec y = core::Vec::Zero(sys.nOwned());
        const solv::KrylovResult g = solv::Gmres(30, 1e-8, 1000).solve(A, M, b, y);
        test::check(r.converged && !r.breakdown && t2 <= 1.1e-8 && std::abs(t2 - r.true_rel_residual) < 1e-14 &&
                        Mc.count == r.iterations && r.history.size() == static_cast<std::size_t>(r.iterations) + 1 &&
                        r.iterations <= 2 * g.iterations + 4,
                    label + ", local " + approx + ": BiCGStab " + std::to_string(r.iterations) + " applications of M (" +
                        std::to_string(r.restarts) + " restarts), true residual " + test::sci(t2) + "; GMRES " +
                        std::to_string(g.iterations) + " it.",
                    comm);
    }
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    try {
        // 1. Convergence.
        checkConvergence(bubbles(), "bubbles3d 24^3 (symmetric), RAS/deflated");
        {
            core::SolverConfig c;
            c.generate = "convdiff3d";
            c.nx = c.ny = c.nz = 16;
            c.nu = 0.01;
            c.rhs = "ones";
            c.tau = 0.3;
            c.nev = 16;
            c.dense_threshold = 0;
            c.coarse_solver = "root";
            checkConvergence(c, "convdiff3d 16^3, nu 0.01 (nonsymmetric), RAS/deflated");
        }

        core::SolverConfig cfg = bubbles();
        cfg.local_approx = "amg";
        part::DecompositionInfo info;
        const core::LocalSystem sys = part::DomainDecomposer::decompose(cfg, comm, info);
        part::Subdomain sd(sys, cfg.pou);
        linalg::DistributedMatrix A(sd);
        const prec::TwoLevelSchwarz M(sd, A, cfg);
        const core::Vec b = Eigen::Map<const core::Vec>(sys.rhs.data(), sys.nOwned());
        const double b1 = norm1(b, comm);

        // 2. OpenFOAM stopping test: first application below relTol (half or full step).
        for (const double rtol : {1e-1, 1e-2, 1e-4, 1e-8}) {
            core::Vec x = core::Vec::Zero(sys.nOwned());
            const solv::KrylovResult r = solv::BiCgStab(rtol, 500, solv::StopNorm::OpenFOAM).solve(A, M, b, x);
            const double t1 = norm1(residual(A, b, x), comm) / b1;
            double before = 1.0; // true residual one application earlier
            if (r.iterations > 1) {
                core::Vec y = core::Vec::Zero(sys.nOwned());
                solv::BiCgStab(rtol, r.iterations - 1, solv::StopNorm::OpenFOAM).solve(A, M, b, y);
                before = norm1(residual(A, b, y), comm) / b1;
            }
            test::check(r.converged && r.restarts == 0 && t1 <= rtol * (1 + 1e-10) && before > rtol &&
                            std::abs(r.initial_residual_of - 1.0) < 1e-12 &&
                            std::abs(r.final_residual_of - t1) < 1e-12 * std::max(1.0, t1 / rtol),
                        "OpenFOAM norm, relTol " + test::sci(rtol) + ": " + std::to_string(r.iterations) +
                            " applications (" + (r.iterations % 2 ? "half step" : "full step") + "), sum|r|/sum|b| " +
                            test::sci(t1) + " (one application earlier " + test::sci(before) + ")",
                        comm);
        }
        {
            core::Vec x(sys.nOwned());
            for (Eigen::Index i = 0; i < x.size(); ++i) x[i] = 0.7 + 0.3 * std::sin(0.37 * static_cast<double>(i + sys.rowBegin()));
            double sc[2] = {x.sum(), static_cast<double>(x.size())};
            MPI_Allreduce(MPI_IN_PLACE, sc, 2, MPI_DOUBLE, MPI_SUM, comm);
            const double xm = sc[0] / sc[1];
            core::Vec A1, Ax0;
            A.apply(core::Vec::Ones(sys.nOwned()), A1);
            A.apply(x, Ax0);
            const double F = norm1(Ax0 - xm * A1, comm) + norm1(b - xm * A1, comm) + 1e-20;
            const double l10 = norm1(b - Ax0, comm);
            const solv::KrylovResult r = solv::BiCgStab(1e-3, 500, solv::StopNorm::OpenFOAM).solve(A, M, b, x);
            const double t1 = norm1(residual(A, b, x), comm);
            test::check(r.converged && std::abs(r.initial_residual_of - l10 / F) < 1e-12 * (l10 / F) &&
                            std::abs(r.final_residual_of - t1 / F) < 1e-9 * (l10 / F) && t1 <= 1e-3 * l10 * (1 + 1e-10),
                        "OpenFOAM normalisation, x0 = 0.7 + 0.3 sin: initial residual " + test::sci(r.initial_residual_of) +
                            " (expected " + test::sci(l10 / F) + "), relTol 1e-3 reached (" + test::sci(t1 / l10) + ")",
                        comm);
        }

        // 3. 2-norm test, no preconditioner, budget of applications.
        {
            core::Vec x = core::Vec::Zero(sys.nOwned());
            const solv::KrylovResult r = solv::BiCgStab(1e-8, 500).solve(A, M, b, x);
            const double t2 = linalg::norm2(residual(A, b, x), comm) / linalg::norm2(b, comm);
            test::check(r.converged && t2 <= 1.1e-8 && r.initial_residual_of == 0.0 && std::abs(t2 - r.true_rel_residual) < 1e-14,
                        "2-norm test: " + std::to_string(r.iterations) + " applications, true residual " + test::sci(t2), comm);

            core::SolverConfig c;
            c.generate = "laplace3d";
            c.nx = c.ny = c.nz = 12;
            part::DecompositionInfo li;
            const core::LocalSystem ls = part::DomainDecomposer::decompose(c, comm, li);
            part::Subdomain lsd(ls, c.pou);
            linalg::DistributedMatrix LA(lsd);
            core::Vec lb(ls.nOwned()); // generic rhs (the all-ones vector excites few modes of this generator)
            for (Eigen::Index i = 0; i < lb.size(); ++i) lb[i] = std::sin(0.37 * static_cast<double>(i + ls.rowBegin())) + 0.3;
            prec::IdentityPreconditioner I;
            core::Vec lx = core::Vec::Zero(ls.nOwned());
            const solv::KrylovResult ri = solv::BiCgStab(1e-8, 2000).solve(LA, I, lb, lx);
            test::check(ri.converged && ri.true_rel_residual <= 1.1e-8,
                        "unpreconditioned BiCGStab, laplace3d 12^3: " + std::to_string(ri.iterations) + " applications", comm);

            core::Vec xb = core::Vec::Zero(sys.nOwned());
            const solv::KrylovResult rb = solv::BiCgStab(1e-12, 3).solve(A, M, b, xb);
            const double tb = linalg::norm2(residual(A, b, xb), comm) / linalg::norm2(b, comm);
            test::check(!rb.converged && rb.iterations == 3 && std::abs(tb - rb.true_rel_residual) < 1e-14 && tb < 1.0,
                        "budget of 3 applications: stops at the half step of step 2 with a consistent x (true residual " +
                            test::sci(tb) + ")",
                        comm);
        }

        // 4. Time loop.
        for (const std::string reuse : {"auto", "off"}) {
            core::SolverConfig c = bubbles();
            c.nx = c.ny = c.nz = 16;
            c.nbubbles = 6;
            c.rhs = "random";
            c.steps = 3;
            c.correctors = 2;
            c.reuse = reuse;
            c.krylov = "bicgstab";
            c.local_approx = "amg";
            c.stop_norm = "openfoam";
            c.rtol = 1e-2;
            const auto recs = solv::runTimeLoop(c, comm, utils::Log::nullStream());
            bool ok = recs.size() == 6;
            int its = 0;
            for (const auto& rr : recs) {
                ok = ok && rr.converged;
                its += rr.iterations;
            }
            test::check(ok, "time loop, BiCGStab + RAS/deflated (AMG), reuse " + reuse + ": 6 solves converged, " +
                                std::to_string(its) + " applications in total",
                        comm);
        }

        // 5. Configuration.
        {
            auto parse = [](std::vector<std::string> args) {
                args.insert(args.begin(), {"prog", "--generate", "laplace2d"});
                std::vector<char*> av;
                for (auto& a : args) av.push_back(a.data());
                return core::SolverConfig::fromCommandLine(static_cast<int>(av.size()), av.data());
            };
            auto rejected = [&](std::vector<std::string> args) {
                try {
                    parse(std::move(args));
                } catch (const std::invalid_argument&) {
                    return true;
                }
                return false;
            };
            const core::SolverConfig d = parse({});
            const core::SolverConfig c1 = parse({"--krylov", "bicgstab"});
            const bool ok = d.krylov == "gmres" && c1.krylov == "bicgstab" && rejected({"--krylov", "cg"}) &&
                            solv::makeKrylovSolver(d)->name() == "GMRES(30)" &&
                            solv::makeKrylovSolver(c1)->name() == "BiCGStab";
            test::check(ok, "configuration: --krylov gmres|bicgstab (default gmres), unknown value rejected", comm);
        }
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int f = test::failures();
    if (core::mpiRank(MPI_COMM_WORLD) == 0) std::cout << (f ? "FAILED" : "ALL PASSED") << std::endl;
    MPI_Finalize();
    return f ? 1 : 0;
}
