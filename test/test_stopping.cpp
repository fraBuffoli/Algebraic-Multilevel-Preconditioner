/**
 * @file test_stopping.cpp
 * @brief GMRES stopping tests: the OpenFOAM normalised residual (StopNorm::OpenFOAM) and the
 *        2-norm test (StopNorm::L2).
 *
 *  - OpenFOAM norm: GMRES stops at the first iteration whose true residual satisfies
 *    sum|b - A x| <= rtol sum|b - A x0| (checked against the true residual of the solution and of
 *    the solution one iteration earlier), for several tolerances and restart lengths;
 *  - normalisation factor of OpenFOAM (lduMatrix::solver::normFactor) for a nonzero initial guess;
 *  - 2-norm test unchanged; configuration: --stop-norm l2|openfoam.
 */
#include "config.hpp"
#include "distributed_matrix.hpp"
#include "domain_decomposer.hpp"
#include "gmres.hpp"
#include "subdomain.hpp"
#include "test_utils.hpp"
#include "vector_ops.hpp"
#include "two_level_schwarz.hpp"

#include <cmath>
#include <iostream>

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

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    try {
        core::SolverConfig cfg;
        cfg.generate = "bubbles3d";
        cfg.nx = cfg.ny = cfg.nz = 24;
        cfg.nbubbles = 20;
        cfg.radius = 0.1;
        cfg.rhs = "generated";
        cfg.tau = 0.3;
        cfg.nev = 8;
        cfg.coarse_solver = "root";
        part::DecompositionInfo info;
        const core::LocalSystem sys = part::DomainDecomposer::decompose(cfg, comm, info);
        part::Subdomain sd(sys, cfg.pou);
        linalg::DistributedMatrix A(sd);
        prec::TwoLevelSchwarz M(sd, A, cfg);
        const core::Vec b = Eigen::Map<const core::Vec>(sys.rhs.data(), sys.nOwned());
        const double b1 = norm1(b, comm);

        // 1. OpenFOAM norm from x0 = 0: stops exactly at the first iteration below rtol.
        for (const double rtol : {1e-1, 1e-2, 1e-4, 1e-8})
            for (const int restart : {30, 4}) {
                core::Vec x = core::Vec::Zero(sys.nOwned());
                const solv::GmresResult r = solv::Gmres(restart, rtol, 500, solv::StopNorm::OpenFOAM).solve(A, M, b, x);
                const double t1 = norm1(residual(A, b, x), comm) / b1;
                double before = 1.0; // true residual one iteration earlier
                if (r.iterations > 1) {
                    core::Vec y = core::Vec::Zero(sys.nOwned());
                    solv::Gmres(restart, rtol, r.iterations - 1, solv::StopNorm::OpenFOAM).solve(A, M, b, y);
                    before = norm1(residual(A, b, y), comm) / b1;
                }
                test::check(r.converged && t1 <= rtol * (1 + 1e-10) && before > rtol && std::abs(r.initial_residual_of - 1.0) < 1e-12 &&
                                std::abs(r.final_residual_of - t1) < 1e-12 * std::max(1.0, t1 / rtol),
                            "OpenFOAM norm, rtol " + test::sci(rtol) + ", GMRES(" + std::to_string(restart) + "): " +
                                std::to_string(r.iterations) + " it., sum|r|/sum|b| " + test::sci(t1) + " (one iteration earlier " +
                                test::sci(before) + "), 2-norm " + test::sci(r.true_rel_residual),
                            comm);
            }

        // 2. Normalisation factor of OpenFOAM for a nonzero x0: F = sum|A x0 - xm A 1| + sum|b - xm A 1|,
        //    xm = mean of x0.
        {
            core::Vec x(sys.nOwned());
            for (Eigen::Index i = 0; i < x.size(); ++i) x[i] = 0.7 + 0.3 * std::sin(0.37 * static_cast<double>(i + sys.rowBegin()));
            double sc[2] = {x.sum(), static_cast<double>(x.size())};
            MPI_Allreduce(MPI_IN_PLACE, sc, 2, MPI_DOUBLE, MPI_SUM, comm);
            const double c = sc[0] / sc[1];
            const core::Vec r0 = residual(A, b, x);
            core::Vec A1;
            A.apply(core::Vec::Ones(sys.nOwned()), A1);
            core::Vec Ax0;
            A.apply(x, Ax0);
            const double F = norm1(Ax0 - c * A1, comm) + norm1(b - c * A1, comm) + 1e-20;
            const double l10 = norm1(r0, comm);
            const solv::GmresResult r = solv::Gmres(30, 1e-3, 500, solv::StopNorm::OpenFOAM).solve(A, M, b, x);
            const double t1 = norm1(residual(A, b, x), comm);
            test::check(r.converged && std::abs(r.initial_residual_of - l10 / F) < 1e-12 * (l10 / F) &&
                            std::abs(r.final_residual_of - t1 / F) < 1e-9 * (l10 / F) && t1 <= 1e-3 * l10 * (1 + 1e-10),
                        "OpenFOAM normalisation, x0 = 0.7 + 0.3 sin: initial residual " + test::sci(r.initial_residual_of) + " (expected " +
                            test::sci(l10 / F) + "), relTol 1e-3 reached (" + test::sci(t1 / l10) + ")",
                        comm);
        }

        // 3. 2-norm test (default) unchanged.
        {
            core::Vec x = core::Vec::Zero(sys.nOwned());
            const solv::GmresResult r = solv::Gmres(30, 1e-8, 500).solve(A, M, b, x);
            const double t2 = linalg::norm2(residual(A, b, x), comm) / linalg::norm2(b, comm);
            test::check(r.converged && t2 <= 1.1e-8 && r.initial_residual_of == 0.0 && std::abs(t2 - r.true_rel_residual) < 1e-12,
                        "2-norm test: " + std::to_string(r.iterations) + " it., true residual " + test::sci(t2), comm);
        }

        // 4. Configuration.
        {
            std::vector<std::string> args{"prog", "--generate", "laplace2d", "--stop-norm", "openfoam"};
            std::vector<char*> av;
            for (auto& a : args) av.push_back(a.data());
            const auto c = core::SolverConfig::fromCommandLine(static_cast<int>(av.size()), av.data());
            bool rejected = false;
            try {
                std::vector<std::string> bad{"prog", "--generate", "laplace2d", "--stop-norm", "l1"};
                std::vector<char*> bv;
                for (auto& a : bad) bv.push_back(a.data());
                core::SolverConfig::fromCommandLine(static_cast<int>(bv.size()), bv.data());
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            test::check(c.stop_norm == "openfoam" && core::SolverConfig().stop_norm == "l2" && rejected &&
                            solv::parseStopNorm("openfoam") == solv::StopNorm::OpenFOAM,
                        "configuration: --stop-norm l2|openfoam (default l2), unknown value rejected", comm);
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
