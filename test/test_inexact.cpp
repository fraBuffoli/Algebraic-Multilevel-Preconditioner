/**
 * @file test_inexact.cpp
 * @brief Tests of the approximate local solver (P7): one AMG V-cycle (AMGCL) instead of A_ii^{-1}.
 *
 *  1. Sequential, on SPD bubbles3d matrices (contrast 1000) and on a nonsymmetric convdiff3d
 *     matrix, for both coarsenings (sa, rs), with and without hierarchy reuse, in double and
 *     single precision:
 *     a. solve() is a fixed linear operator: B(b1 + 2 b2) = B b1 + 2 B b2, and two solves of
 *        the same right-hand side are bitwise identical (required by plain GMRES);
 *     b. quality: iterations of CG (SPD) or GMRES (nonsymmetric: at least 3x fewer than without
 *        preconditioner) preconditioned with B, after factorize() and after refactorize() with
 *        the values of a later time step;
 *     c. `coarse_enough` bounds the coarsest level (more levels with 500 than with 3000);
 *     d. name() reports what was built; unknown or removed options throw.
 *  2. Distributed (two-level, bubbles3d): with `--local-approx amg` the coarse space is the same
 *     as with exact factors (A~_ii and A_00 are not touched), A~_ii is still factorized exactly,
 *     GMRES reaches the tolerance on the *true* residual, and the iterations stay within 2x + 5
 *     those of the exact local solves; one-level RAS with AMG converges.
 *  3. Time loop with AMG and refactorizations (full setup and reused hierarchy): levels, convergence.
 *  4. Configuration: the --local-approx / --amg-* options are parsed and validated, defaults.
 *
 * In a build without AMGCL only the rejection of `--local-approx amg` is checked.
 * Run e.g. `mpirun -np 4 ./bin/test_inexact` (a few seconds).
 */
#include "config.hpp"
#include "distributed_matrix.hpp"
#include "domain_decomposer.hpp"
#include "gmres.hpp"
#include "inexact_solver.hpp"
#include "local_solver.hpp"
#include "logger.hpp"
#include "matrix_generators.hpp"
#include "one_level_schwarz.hpp"
#include "subdomain.hpp"
#include "test_utils.hpp"
#include "time_loop.hpp"
#include "two_level_schwarz.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <regex>
#include <stdexcept>

using namespace schwarz2lvl;

namespace {

core::SpMat generated(const std::string& gen, int nx, double time = 0.0)
{
    utils::GeneratorOptions o;
    o.nx = o.ny = o.nz = nx;
    o.nu = 0.05;
    o.nbubbles = 6;
    o.radius = 0.15;
    o.rho_ratio = 1000.0;
    o.pulse = 0.5; // convdiff: the convection changes with the time step
    o.pulse_period = 10;
    o.time = time;
    return test::toEigen(utils::MatrixGenerator::generate(gen, o).A);
}

/// @brief CG preconditioned with B: iterations to reduce ||r|| by @p tol (maxit + 1 if not reached).
int pcg(const core::SpMat& A, const linalg::LocalSolver& B, const core::Vec& b, double tol, int maxit)
{
    core::Vec x = core::Vec::Zero(b.size()), r = b, z, p, q;
    B.solve(r, z);
    p = z;
    double rz = r.dot(z);
    const double nb = b.norm();
    for (int k = 1; k <= maxit; ++k) {
        q = A * p;
        const double alpha = rz / p.dot(q);
        x += alpha * p;
        r -= alpha * q;
        if (r.norm() <= tol * nb) return k;
        B.solve(r, z);
        const double rz1 = r.dot(z);
        p = z + (rz1 / rz) * p;
        rz = rz1;
    }
    return maxit + 1;
}

/// @brief Right-preconditioned GMRES without restart (x0 = 0): iterations to reduce ||b - A x|| by
///        @p tol (maxit + 1 if not reached). @p B applies the preconditioner (z = B v).
template <typename Apply> int gmres(const core::SpMat& A, Apply&& B, const core::Vec& b, double tol, int maxit)
{
    const Eigen::Index n = b.size();
    core::Mat V(n, maxit + 1), H = core::Mat::Zero(maxit + 1, maxit);
    const double beta = b.norm();
    V.col(0) = b / beta;
    core::Vec z, w;
    for (int k = 0; k < maxit; ++k) {
        B(core::Vec(V.col(k)), z);
        w = A * z;
        for (int j = 0; j <= k; ++j) { // modified Gram-Schmidt
            H(j, k) = V.col(j).dot(w);
            w -= H(j, k) * V.col(j);
        }
        H(k + 1, k) = w.norm();
        core::Vec e1 = core::Vec::Zero(k + 2);
        e1[0] = beta;
        const core::Mat Hk = H.topLeftCorner(k + 2, k + 1);
        const core::Vec y = Hk.colPivHouseholderQr().solve(e1);
        if ((e1 - Hk * y).norm() <= tol * beta || H(k + 1, k) == 0.0) return k + 1;
        V.col(k + 1) = w / H(k + 1, k);
    }
    return maxit + 1;
}

linalg::LocalSolverSpec amgSpec(const std::string& coarsening, bool reuse, bool symmetric, bool single,
                                int coarse_enough = 500)
{
    linalg::LocalSolverSpec s;
    s.approx = "amg";
    s.symmetric = symmetric;
    s.single = single;
    s.amg.coarsening = coarsening;
    s.amg.reuse = reuse;
    s.amg.coarse_enough = coarse_enough;
    return s;
}

/// @brief Number of AMG levels reported in the name ("[3 levels, ...]"), -1 if absent.
int levelsOf(const std::string& name)
{
    std::smatch m;
    return std::regex_search(name, m, std::regex(R"(\[(\d+) levels)")) ? std::stoi(m[1]) : -1;
}

} // namespace

// --------------------------------------------------------------------------- 1. sequential
void checkLocalSpd()
{
    const core::SpMat A0 = generated("bubbles3d", 20), A1 = generated("bubbles3d", 20, 3.0);
    const core::Vec b1 = core::Vec::LinSpaced(A0.rows(), -1.0, 2.0).array().sin();
    const core::Vec b2 = core::Vec::LinSpaced(A0.rows(), 0.0, 7.0).array().cos();
    for (const std::string co : {"sa", "rs"})
        for (const bool reuse : {false, true})
            for (const bool single : {false, true}) {
                auto S = linalg::makeLocalSolver(amgSpec(co, reuse, true, single));
                const bool ok0 = S->factorize(A0);
                core::Vec x1, x2, x12, y1;
                double lin = 1.0;
                bool same = false;
                int it0 = -1, it1 = -1;
                if (ok0) {
                    S->solve(b1, x1);
                    S->solve(b2, x2);
                    S->solve(core::Vec(b1 + 2.0 * b2), x12);
                    S->solve(b1, y1);
                    lin = (x12 - x1 - 2.0 * x2).norm() / x12.norm();
                    same = (y1.array() == x1.array()).all();
                    it0 = pcg(A0, *S, b1, 1e-8, 400);
                }
                const bool ok1 = ok0 && S->refactorize(A1);
                if (ok1) it1 = pcg(A1, *S, b1, 1e-8, 400);
                // float: B is linear up to rounding of the right-hand side (relative 1e-7 per entry).
                const double lin_tol = single ? 1e-5 : 1e-12;
                const int max_it = single ? 25 : 20; // measured: 8-13 iterations
                const std::string nm = S->name();
                const bool name_ok = nm.rfind("amg " + co + "+gs", 0) == 0 &&
                                     (nm.find("single") != std::string::npos) == single &&
                                     (nm.find("reused hierarchy") != std::string::npos) == reuse;
                test::check(ok0 && ok1 && lin < lin_tol && same && it0 <= max_it && it1 <= max_it && name_ok,
                            "SPD bubbles3d 20^3: " + nm + ": linear (" + test::sci(lin) + "), deterministic, PCG " +
                                std::to_string(it0) + " / " + std::to_string(it1) + " it. after refactorize (<= " +
                                std::to_string(max_it) + ")");
            }
    // coarse_enough: the coarsest level has at most that many unknowns (here 8000 unknowns).
    auto S500 = linalg::makeLocalSolver(amgSpec("sa", false, true, false, 500));
    auto S3000 = linalg::makeLocalSolver(amgSpec("sa", false, true, false, 3000));
    const bool ok = S500->factorize(A0) && S3000->factorize(A0);
    const int l500 = levelsOf(S500->name()), l3000 = levelsOf(S3000->name());
    test::check(ok && l500 > l3000 && l3000 >= 2,
                "coarse_enough 500 -> " + std::to_string(l500) + " levels, 3000 -> " + std::to_string(l3000) + " levels");
}

void checkLocalNonsymmetric()
{
    const core::SpMat A0 = generated("convdiff3d", 16), A1 = generated("convdiff3d", 16, 4.0);
    const core::Vec b = core::Vec::LinSpaced(A0.rows(), -1.0, 2.0).array().sin();
    const int none = gmres(A0, [](const core::Vec& v, core::Vec& z) { z = v; }, b, 1e-8, 300);
    for (const std::string co : {"sa", "rs"})
        for (const bool single : {false, true}) {
            auto S = linalg::makeLocalSolver(amgSpec(co, false, false, single));
            auto B = [&S](const core::Vec& v, core::Vec& z) { S->solve(v, z); };
            const bool ok0 = S->factorize(A0);
            const int it0 = ok0 ? gmres(A0, B, b, 1e-8, 300) : 999;
            const bool ok1 = ok0 && S->refactorize(A1);
            const int it1 = ok1 ? gmres(A1, B, b, 1e-8, 300) : 999;
            test::check(ok0 && ok1 && 3 * it0 <= none && 3 * it1 <= none,
                        "nonsymmetric convdiff3d 16^3: " + S->name() + ": GMRES " + std::to_string(it0) + " / " +
                            std::to_string(it1) + " it. after refactorize (unpreconditioned " + std::to_string(none) +
                            ")");
        }
}

void checkErrors()
{
    auto throws = [](linalg::LocalSolverSpec s) {
        try {
            linalg::makeLocalSolver(s);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    linalg::LocalSolverSpec s;
    s.approx = "ic"; // removed after the first P7 campaign
    bool ok = throws(s);
    s.approx = "amg";
    if (linalg::amgAvailable()) {
        s.amg.coarsening = "foo";
        ok = ok && throws(s);
        s.amg.coarsening = "sa";
        s.amg.sweeps = 0;
        ok = ok && throws(s);
        s.amg.sweeps = 1;
        s.amg.coarse_enough = 0;
        ok = ok && throws(s);
    } else {
        ok = ok && throws(s); // amg requested in a build without AMGCL
    }
    const auto av = linalg::availableLocalApprox();
    ok = ok && av.front() == "none" && (std::find(av.begin(), av.end(), "amg") != av.end()) == linalg::amgAvailable() &&
         av.size() == (linalg::amgAvailable() ? 2u : 1u);
    test::check(ok, std::string("unknown approximations / AMG options are rejected (AMGCL ") +
                        (linalg::amgAvailable() ? "compiled in)" : "not compiled in)"));
}

// --------------------------------------------------------------------------- 2. distributed
void checkDistributed()
{
    MPI_Comm comm = MPI_COMM_WORLD;
    core::SolverConfig cfg;
    cfg.generate = "bubbles3d";
    cfg.nx = cfg.ny = cfg.nz = 32; // subdomains of >= 4k unknowns up to 8 processes: AMG has >= 2 levels
    cfg.nbubbles = 8;
    cfg.radius = 0.12;
    cfg.tau = 0.3;
    cfg.nev = 8;
    cfg.dense_threshold = 0;
    cfg.coarse_solver = "root";
    part::DecompositionInfo info;
    const core::LocalSystem sys = part::DomainDecomposer::decompose(cfg, comm, info);
    part::Subdomain sd(sys, cfg.pou);
    linalg::DistributedMatrix A(sd);
    const core::Vec b = Eigen::Map<const core::Vec>(sys.rhs.data(), sd.nOwned());

    auto gmresSolve = [&](const prec::Preconditioner& M) {
        core::Vec x = core::Vec::Zero(sd.nOwned());
        return solv::Gmres(30, 1e-8, 500).solve(A, M, b, x);
    };
    prec::TwoLevelSchwarz Mex(sd, A, cfg);
    const solv::GmresResult rex = gmresSolve(Mex);
    const core::GlobalIndex n0 = Mex.coarse().dimension();
    // One process: the exact local solve is a direct solve of A (1 iteration), hence max(., 10).
    const int ref = std::max(rex.iterations, 10), bound = 2 * ref + 5;

    for (const std::string co : {"sa", "rs"})
        for (const std::string prec : {"double", "single"}) {
            core::SolverConfig c = cfg;
            c.local_approx = "amg";
            c.amg_coarsening = co;
            c.local_precision = prec;
            prec::TwoLevelSchwarz M(sd, A, c);
            const solv::GmresResult r = gmresSolve(M);
            const auto& st = M.stats();
            test::check(r.converged && r.true_rel_residual <= 1.1e-8 && r.iterations <= bound &&
                            M.coarse().dimension() == n0 && st.local_factorization.rfind("amg " + co, 0) == 0 &&
                            st.atilde_factorization.find("cholesky") != std::string::npos,
                        "two-level, A_ii " + st.local_factorization + ": GMRES " + std::to_string(r.iterations) +
                            " it. (exact " + std::to_string(rex.iterations) + ", bound " + std::to_string(bound) +
                            "), true residual " + test::sci(r.true_rel_residual) + ", same n0 = " +
                            std::to_string(M.coarse().dimension()) + ", A~_ii " + st.atilde_factorization,
                        comm);
        }

    core::SolverConfig c = cfg;
    c.local_approx = "amg";
    prec::OneLevelSchwarz M1ex(sd, prec::OneLevelSchwarz::Variant::RAS, prec::localSolverSpec(cfg, sys.symmetric));
    prec::OneLevelSchwarz M1(sd, prec::OneLevelSchwarz::Variant::RAS, prec::localSolverSpec(c, sys.symmetric));
    const solv::GmresResult r1ex = gmresSolve(M1ex), r1 = gmresSolve(M1);
    test::check(r1.converged && r1.true_rel_residual <= 1.1e-8 && r1.iterations <= 2 * std::max(r1ex.iterations, 10) + 5,
                "one-level RAS, A_ii " + M1.localSolver().name() + ": GMRES " + std::to_string(r1.iterations) +
                    " it. (exact " + std::to_string(r1ex.iterations) + ")",
                comm);
}

// --------------------------------------------------------------------------- 3. time loop
void checkTimeLoop()
{
    for (const bool reuse : {false, true}) {
        core::SolverConfig cfg;
        cfg.generate = "bubbles3d";
        cfg.nx = cfg.ny = cfg.nz = 16;
        cfg.nbubbles = 6;
        cfg.radius = 0.15;
        cfg.cfl = 1.0;
        cfg.tau = 0.3;
        cfg.nev = 8;
        cfg.steps = 3;
        cfg.correctors = 2;
        cfg.reuse = "refactor";
        cfg.local_approx = "amg";
        cfg.amg_reuse = reuse;
        const auto r = solv::runTimeLoop(cfg, MPI_COMM_WORLD, utils::Log::nullStream());
        std::string lv;
        for (const auto& s : r) lv += std::to_string(static_cast<int>(s.level));
        const bool conv = std::all_of(r.begin(), r.end(), [](const solv::SolveRecord& s) {
            return s.converged && s.true_rel_residual <= 1.1e-8;
        });
        test::check(lv == "201010" && conv, std::string("time loop, AMG ") +
                                                (reuse ? "with reused hierarchy" : "with full setups") +
                                                " + refactorizations: levels " + lv + ", all converged");
    }
}

// --------------------------------------------------------------------------- 4. configuration
void checkConfig()
{
    auto parse = [](std::vector<std::string> args) {
        args.insert(args.begin(), {"prog", "--generate", "laplace2d"});
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(a.data());
        return core::SolverConfig::fromCommandLine(static_cast<int>(argv.size()), argv.data());
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
    const bool defaults = d.local_approx == "none" && d.amg_coarsening == "sa" && d.amg_sweeps == 1 &&
                          d.amg_coarse_enough == 500 && !d.amg_reuse && d.rebuild_growth == 4.0;
    const core::SolverConfig c = parse({"--local-approx", "amg", "--amg-coarsening", "rs", "--amg-sweeps", "2",
                                        "--amg-coarse-enough", "200", "--amg-reuse", "1"});
    const auto s = prec::localSolverSpec(c, true);
    const bool ok = defaults && s.approx == "amg" && s.amg.coarsening == "rs" && s.amg.sweeps == 2 &&
                    s.amg.coarse_enough == 200 && s.amg.reuse && rejected({"--local-approx", "ic"}) &&
                    rejected({"--amg-relax", "gs"}) && rejected({"--amg-coarsening", "aggr"}) &&
                    rejected({"--amg-sweeps", "0"}) && rejected({"--amg-coarse-enough", "0"}) &&
                    rejected({"--rebuild-growth", "-1"});
    test::check(ok, "configuration: defaults (sa, 1 sweep, coarsest <= 500, no reuse, --rebuild-growth 4), "
                    "options parsed into the A_ii specification, bad or removed options rejected");
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    try {
        checkErrors();
        checkConfig();
        if (linalg::amgAvailable()) {
            checkLocalSpd();
            checkLocalNonsymmetric();
            checkDistributed();
            checkTimeLoop();
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
