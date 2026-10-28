/**
 * @file test_warmstart.cpp
 * @brief Pieces of the warm-start campaign (test/gamg/Allrun_warm.sh): the p_rgh source of the
 *        bubbles generators, the absolute tolerance of the OpenFOAM stopping test, the warm start
 *        and the tolerances of the last corrector in the time loop.
 *
 *  - --bubble-source buoyancy: right-hand side = -div(phig) of a VoF solver for the fluid at rest,
 *    checked against a reference written face by face from the bubble field (3D and 2D); the matrix
 *    is the one of --bubble-source alpha; the sum over every bubble (connected set of gas cells and
 *    their neighbours) is zero, and so is the total;
 *  - --atol (OpenFOAM `tolerance`), GMRES and BiCGStab: with rtol 0 the solve stops at the first
 *    iterate with sum|b - A x| / F <= atol (F from x0, recomputed here); with both, at
 *    max(rtol sum|r0|, atol F); no iteration at all when x0 already satisfies the test;
 *  - time loop with --warm-start and --final-rtol 0 --final-atol: every solve after the first starts
 *    from the previous solution (initial normalised residual < 1, = 1 from 0), the correctors stop
 *    at relTol, the last corrector of every step at the absolute tolerance, and the warm start needs
 *    fewer applications than the cold start for the same tolerances;
 *  - reuse policy: with kinds of solves (corrector index) every kind has its own reference q_ref, so
 *    that solves of different kinds with a fresh basis waste nothing;
 *  - configuration: --atol only with --stop-norm openfoam, rtol 0 only with an atol, --bubble-source.
 */
#include "bicgstab.hpp"
#include "bubble_field.hpp"
#include "config.hpp"
#include "distributed_matrix.hpp"
#include "domain_decomposer.hpp"
#include "gmres.hpp"
#include "krylov.hpp"
#include "logger.hpp"
#include "matrix_generators.hpp"
#include "reuse_policy.hpp"
#include "subdomain.hpp"
#include "test_utils.hpp"
#include "time_loop.hpp"
#include "two_level_schwarz.hpp"
#include "vector_ops.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

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

/// @brief OpenFOAM normalisation factor of x0, written out (lduMatrix::solver::normFactor).
double normFactor(const linalg::DistributedMatrix& A, const core::Vec& b, const core::Vec& x0, MPI_Comm comm)
{
    double sc[2] = {x0.sum(), static_cast<double>(x0.size())};
    MPI_Allreduce(MPI_IN_PLACE, sc, 2, MPI_DOUBLE, MPI_SUM, comm);
    const double xm = sc[0] / sc[1];
    core::Vec A1, Ax0;
    A.apply(core::Vec::Ones(x0.size()), A1);
    A.apply(x0, Ax0);
    return norm1(Ax0 - xm * A1, comm) + norm1(b - xm * A1, comm) + 1e-20;
}

/// @brief Buoyancy source of the generator against a face-by-face reference (sequential).
void checkBuoyancy(int dim, int n, int nb, double R, double time)
{
    utils::GeneratorOptions o;
    o.nx = o.ny = n;
    o.nz = (dim == 3) ? n : 1;
    o.nbubbles = nb;
    o.radius = R;
    o.cfl = 1.0;
    o.time = time;
    o.rho_ratio = 1000.0;
    const std::string name = (dim == 3) ? "bubbles3d" : "bubbles2d";
    const utils::GeneratedProblem ga = utils::MatrixGenerator::generate(name, o);
    o.source = "buoyancy";
    const utils::GeneratedProblem gb = utils::MatrixGenerator::generate(name, o);

    utils::BubbleFieldParams p;
    p.dim = dim;
    p.nx = p.ny = n;
    p.nz = o.nz;
    p.nbubbles = nb;
    p.radius = R;
    p.cfl = o.cfl;
    p.time = time;
    std::vector<double> alpha;
    utils::BubbleField(p).fill(alpha);
    const int nz = o.nz;
    const double h = 1.0 / n;
    const auto ncell = static_cast<std::size_t>(n) * n * nz;
    auto id = [&](int i, int j, int k) { return static_cast<std::size_t>(i) + n * (static_cast<std::size_t>(j) + static_cast<std::size_t>(n) * k); };
    auto rho = [&](std::size_t c) { return alpha[c] / o.rho_ratio + (1.0 - alpha[c]); };
    // Reference: b_P = sum over the faces of P of beta_f gh_f (rho_nb - rho_P) |S_f| / d, gh_f = 1 - z_f
    // (z = vertical: y in 2D), |S_f| / d = h^(dim-2) on the uniform mesh.
    const double geom = (dim == 3) ? h : 1.0;
    std::vector<double> ref(ncell, 0.0);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const std::size_t P = id(i, j, k);
                const int v = (dim == 3) ? k : j; // vertical index
                auto face = [&](int ii, int jj, int kk, double zf) {
                    if (ii < 0 || jj < 0 || kk < 0 || ii >= n || jj >= n || kk >= nz) return;
                    const std::size_t N = id(ii, jj, kk);
                    const double bf = 0.5 * (1.0 / rho(P) + 1.0 / rho(N));
                    ref[P] += bf * (1.0 - zf) * (rho(N) - rho(P)) * geom;
                };
                const double zc = (v + 0.5) * h;
                face(i - 1, j, k, zc);
                face(i + 1, j, k, zc);
                if (dim == 3) {
                    face(i, j - 1, k, zc);
                    face(i, j + 1, k, zc);
                    face(i, j, k - 1, k * h);
                    face(i, j, k + 1, (k + 1) * h);
                } else {
                    face(i, j - 1, k, j * h);
                    face(i, j + 1, k, (j + 1) * h);
                }
            }
    double err = 0.0, bmax = 0.0, sum = 0.0, abssum = 0.0;
    for (std::size_t c = 0; c < ncell; ++c) {
        err = std::max(err, std::abs(gb.rhs[c] - ref[c]));
        bmax = std::max(bmax, std::abs(ref[c]));
        sum += gb.rhs[c];
        abssum += std::abs(gb.rhs[c]);
    }
    // Bubbles: connected components (faces) of the gas cells (alpha > 0) grown by one layer of
    // neighbours. A face on the boundary of such a set joins two cells without gas: no density jump,
    // no flux, so the sum of the source over the set is zero.
    std::vector<char> mask(ncell, 0);
    std::vector<int> comp(ncell, -1);
    auto neighbours = [&](std::size_t c, std::vector<std::size_t>& out) {
        out.clear();
        const int i = static_cast<int>(c % n), j = static_cast<int>((c / n) % n), k = static_cast<int>(c / (static_cast<std::size_t>(n) * n));
        const int d[6][3] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
        for (const auto& e : d) {
            const int ii = i + e[0], jj = j + e[1], kk = k + e[2];
            if (ii >= 0 && jj >= 0 && kk >= 0 && ii < n && jj < n && kk < nz) out.push_back(id(ii, jj, kk));
        }
    };
    std::vector<std::size_t> nbr;
    for (std::size_t c = 0; c < ncell; ++c)
        if (alpha[c] > 0.0) {
            mask[c] = 1;
            neighbours(c, nbr);
            for (std::size_t q : nbr) mask[q] = 1;
        }
    int ncomp = 0;
    double worst = 0.0;
    for (std::size_t c0 = 0; c0 < ncell; ++c0) {
        if (!mask[c0] || comp[c0] >= 0) continue;
        double s = 0.0, sa = 0.0;
        std::vector<std::size_t> stack{c0};
        comp[c0] = ncomp;
        while (!stack.empty()) {
            const std::size_t c = stack.back();
            stack.pop_back();
            s += gb.rhs[c];
            sa += std::abs(gb.rhs[c]);
            neighbours(c, nbr);
            for (std::size_t q : nbr)
                if (mask[q] && comp[q] < 0) {
                    comp[q] = ncomp;
                    stack.push_back(q);
                }
        }
        if (sa > 0.0) worst = std::max(worst, std::abs(s) / sa);
        ++ncomp;
    }
    // Same matrix as with the alpha source.
    const bool same_matrix = ga.A.row_ptr == gb.A.row_ptr && ga.A.col == gb.A.col && ga.A.val == gb.A.val;
    test::check(err <= 1e-13 * bmax && bmax > 0.0 && std::abs(sum) <= 1e-13 * abssum && worst <= 1e-12 && same_matrix &&
                    gb.description.find("source=buoyancy") != std::string::npos,
                name + " " + std::to_string(n) + ", " + std::to_string(nb) + " bubbles, t = " + std::to_string(static_cast<int>(time)) +
                    ": buoyancy source = face-by-face reference (rel. err " + test::sci(err / bmax) + "), sum over each of the " +
                    std::to_string(ncomp) + " bubbles ~ 0 (max " + test::sci(worst) + " of sum|b|), total " +
                    test::sci(std::abs(sum) / abssum) + ", same matrix as the alpha source");
}

core::SolverConfig bubbles()
{
    core::SolverConfig c;
    c.generate = "bubbles3d";
    c.nx = c.ny = c.nz = 24;
    c.nbubbles = 20;
    c.radius = 0.1;
    c.rhs = "generated";
    c.bubble_source = "buoyancy";
    c.tau = 0.3;
    c.nev = 8;
    c.coarse_solver = "root";
    return c;
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    try {
        // 1. Buoyancy source of the generators.
        checkBuoyancy(3, 24, 4, 0.08, 0.0); // separate bubbles
        checkBuoyancy(3, 20, 12, 0.12, 13.0); // bubbles crossing the top re-enter at the bottom
        checkBuoyancy(2, 40, 8, 0.08, 5.0);

        // 2.-3. Absolute tolerance of the OpenFOAM test, GMRES and BiCGStab, on the buoyancy system.
        const core::SolverConfig cfg = bubbles();
        part::DecompositionInfo info;
        const core::LocalSystem sys = part::DomainDecomposer::decompose(cfg, comm, info);
        part::Subdomain sd(sys, cfg.pou);
        linalg::DistributedMatrix A(sd);
        prec::TwoLevelSchwarz M(sd, A, cfg);
        const core::Vec b = Eigen::Map<const core::Vec>(sys.rhs.data(), sys.nOwned());
        const double F0 = normFactor(A, b, core::Vec::Zero(sys.nOwned()), comm);
        using Make = std::unique_ptr<solv::KrylovSolver> (*)(double, double, int);
        const std::vector<std::pair<std::string, Make>> solvers = {
            {"GMRES", [](double rt, double at, int it) -> std::unique_ptr<solv::KrylovSolver> {
                 return std::make_unique<solv::Gmres>(30, rt, it, solv::StopNorm::OpenFOAM, at);
             }},
            {"BiCGStab", [](double rt, double at, int it) -> std::unique_ptr<solv::KrylovSolver> {
                 return std::make_unique<solv::BiCgStab>(rt, it, solv::StopNorm::OpenFOAM, at);
             }}};
        for (const auto& [sname, make] : solvers) {
            // 2a. rtol 0, atol only: stops at the first iterate below atol (from x0 = 0: F = sum|b|).
            for (const double atol : {1e-3, 1e-6}) {
                core::Vec x = core::Vec::Zero(sys.nOwned());
                const solv::KrylovResult r = make(0.0, atol, 500)->solve(A, M, b, x);
                const double t1 = norm1(residual(A, b, x), comm) / F0;
                double before = 1.0;
                if (sname == "GMRES" && r.iterations > 1) { // GMRES stops at the first iterate: one earlier is above
                    core::Vec y = core::Vec::Zero(sys.nOwned());
                    make(0.0, atol, r.iterations - 1)->solve(A, M, b, y);
                    before = norm1(residual(A, b, y), comm) / F0;
                }
                test::check(r.converged && r.iterations > 0 && t1 <= 1.1 * atol && before > atol &&
                                std::abs(r.final_residual_of - t1) <= 1e-9 * atol && std::abs(r.initial_residual_of - 1.0) < 1e-12,
                            sname + ", relTol 0, tolerance " + test::sci(atol) + ": " + std::to_string(r.iterations) +
                                " it., sum|r|/F " + test::sci(t1) + (sname == "GMRES" ? " (one iteration earlier " + test::sci(before) + ")" : ""),
                            comm);
            }
            // 2b. Both: the looser of the two decides (max(rtol sum|r0|, atol F)).
            {
                core::Vec x1 = core::Vec::Zero(sys.nOwned()), x2 = core::Vec::Zero(sys.nOwned()), x3 = core::Vec::Zero(sys.nOwned());
                const solv::KrylovResult r_rel = make(1e-4, 0.0, 500)->solve(A, M, b, x1);
                const solv::KrylovResult r_both_tight = make(1e-4, 1e-12, 500)->solve(A, M, b, x2);
                const solv::KrylovResult r_both_loose = make(1e-4, 1e-2, 500)->solve(A, M, b, x3);
                const double t3 = norm1(residual(A, b, x3), comm) / F0;
                // One process: M^{-1} = A^{-1} (one subdomain, exact local solve), so every tolerance is
                // met by the first application and the looser test cannot stop earlier.
                const bool exact = r_rel.iterations <= 1;
                test::check(r_both_tight.iterations == r_rel.iterations &&
                                (exact || r_both_loose.iterations < r_rel.iterations) && t3 <= 1.1e-2 && r_both_loose.converged,
                            sname + ", relTol 1e-4 with tolerance 1e-12 / 1e-2: " + std::to_string(r_both_tight.iterations) + " / " +
                                std::to_string(r_both_loose.iterations) + " it. (relTol alone " + std::to_string(r_rel.iterations) + ")",
                            comm);
            }
            // 2c. Warm start: x0 = an accurate solution, then a slightly perturbed one. F from x0.
            {
                core::Vec xs = core::Vec::Zero(sys.nOwned());
                make(0.0, 1e-10, 1000)->solve(A, M, b, xs);
                core::Vec x = xs;
                const solv::KrylovResult r0 = make(0.0, 1e-6, 500)->solve(A, M, b, x);
                core::Vec xp = xs;
                for (Eigen::Index i = 0; i < xp.size(); ++i)
                    xp[i] *= 1.0 + 1e-3 * std::sin(0.71 * static_cast<double>(i + sys.rowBegin()));
                const double Fp = normFactor(A, b, xp, comm), l1p = norm1(residual(A, b, xp), comm);
                core::Vec y = xp;
                const solv::KrylovResult r1 = make(0.05, 1e-9, 500)->solve(A, M, b, y);
                const double t1 = norm1(residual(A, b, y), comm);
                test::check(r0.converged && r0.iterations == 0 && x == xs && r1.converged && r1.iterations > 0 &&
                                std::abs(r1.initial_residual_of - l1p / Fp) <= 1e-12 * (l1p / Fp) &&
                                t1 <= 1.1 * std::max(0.05 * l1p, 1e-9 * Fp),
                            sname + ", warm start: x0 = solution -> " + std::to_string(r0.iterations) + " it.; x0 perturbed by 1e-3 -> " +
                                std::to_string(r1.iterations) + " it. to relTol 0.05 (initial normalised residual " +
                                test::sci(r1.initial_residual_of) + ")",
                            comm);
            }
        }

        // 4. Time loop: warm start and the last corrector to an absolute tolerance (p_rghFinal).
        for (const std::string krylov : {"bicgstab", "gmres"}) {
            core::SolverConfig c = bubbles();
            c.nx = c.ny = c.nz = 20;
            c.nbubbles = 10;
            c.radius = 0.12;
            c.steps = 4;
            c.correctors = 3;
            c.krylov = krylov;
            c.local_approx = "amg";
            c.stop_norm = "openfoam";
            c.rtol = 0.05;
            c.atol = 1e-7;
            c.final_rtol = 0.0;
            c.final_atol = 1e-7;
            auto run = [&](bool warm) {
                core::SolverConfig cc = c;
                cc.warm_start = warm;
                return solv::runTimeLoop(cc, comm, utils::Log::nullStream());
            };
            const auto warm = run(true), cold = run(false);
            bool ok = warm.size() == 12 && cold.size() == 12;
            int its_warm = 0, its_cold = 0, its_final_warm = 0, its_final_cold = 0;
            double worst_corr = 0.0, worst_final = 0.0, max_init_warm = 0.0, init_cold_err = 0.0;
            for (std::size_t q = 0; ok && q < warm.size(); ++q) {
                const auto& w = warm[q];
                const auto& k = cold[q];
                ok = ok && w.converged && k.converged && w.final_corrector == (w.corrector == 2);
                its_warm += w.iterations;
                its_cold += k.iterations;
                if (w.final_corrector) {
                    its_final_warm += w.iterations;
                    its_final_cold += k.iterations;
                    worst_final = std::max({worst_final, w.final_res_of / 1e-7, k.final_res_of / 1e-7});
                } else {
                    const double thr_w = std::max(0.05 * w.initial_res_of, 1e-7), thr_k = std::max(0.05 * k.initial_res_of, 1e-7);
                    worst_corr = std::max({worst_corr, w.final_res_of / thr_w, k.final_res_of / thr_k});
                }
                if (q > 0) max_init_warm = std::max(max_init_warm, w.initial_res_of);
                init_cold_err = std::max(init_cold_err, std::abs(k.initial_res_of - 1.0));
            }
            ok = ok && worst_corr <= 1.1 && worst_final <= 1.1 && max_init_warm < 1.0 && init_cold_err < 1e-12 &&
                 std::abs(warm.front().initial_res_of - 1.0) < 1e-12 && its_final_warm < its_final_cold;
            test::check(ok, "time loop, " + krylov + ", correctors relTol 0.05, last corrector tolerance 1e-7: all converged, " +
                                "residual / threshold <= " + test::sci(std::max(worst_corr, worst_final)) +
                                "; warm start " + std::to_string(its_warm) + " applications (last correctors " +
                                std::to_string(its_final_warm) + "), from 0 " + std::to_string(its_cold) + " (" +
                                std::to_string(its_final_cold) + "); initial normalised residual after the first solve <= " +
                                test::sci(max_init_warm),
                        comm);
        }

        // 4b. Reuse policy: one reference per kind of solve (corrector index).
        {
            using prec::UpdateLevel;
            auto feed = [](bool kinds) {
                prec::ReusePolicy p(prec::ReusePolicy::Mode::Auto, 4.0, 0);
                p.record(UpdateLevel::Rebuild, 10, 1e-2, true, 2.0, 1.0, 0);        // q = 5 (first corrector)
                p.record(UpdateLevel::Reuse, 30, 1e-5, true, 0.0, 3.0, kinds ? 2 : 0); // q = 6 (last corrector)
                p.record(UpdateLevel::Refactorize, 10, 1e-2, true, 0.2, 1.0, 0);    // as fresh: nothing wasted
                p.record(UpdateLevel::Reuse, 30, 1e-5, true, 0.0, 3.0, kinds ? 2 : 0);
                return p;
            };
            const prec::ReusePolicy with = feed(true), without = feed(false);
            test::check(with.wasted() == 0.0 && std::abs(with.reference(0) - 5.0) < 1e-12 &&
                            std::abs(with.reference(2) - 6.0) < 1e-12 && std::abs(without.wasted() - 1.0) < 1e-12,
                        "reuse policy: q_ref per kind of solve (corrector 0: 5, last corrector: 6 it./decade): nothing "
                        "wasted without aging (one reference for all: " + test::sci(without.wasted()) + " s)",
                        comm);
        }

        // 5. Configuration.
        {
            auto parse = [](std::vector<std::string> args) {
                args.insert(args.begin(), {"prog", "--generate", "bubbles3d"});
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
            const core::SolverConfig c1 = parse({"--stop-norm", "openfoam", "--rtol", "0.05", "--atol", "1e-7", "--final-rtol", "0",
                                                 "--warm-start", "1", "--bubble-source", "buoyancy"});
            const bool ok = d.atol == 0.0 && !d.warm_start && d.final_rtol < 0.0 && d.final_atol < 0.0 && d.bubble_source == "alpha" &&
                            c1.atol == 1e-7 && c1.rtol == 0.05 && c1.final_rtol == 0.0 && c1.warm_start &&
                            c1.bubble_source == "buoyancy" && rejected({"--atol", "1e-7"}) &&
                            rejected({"--stop-norm", "openfoam", "--rtol", "0"}) &&
                            rejected({"--stop-norm", "openfoam", "--final-rtol", "0"}) && rejected({"--bubble-source", "rho"}) &&
                            rejected({"--stop-norm", "openfoam", "--atol", "-1"});
            test::check(ok, "configuration: --atol (openfoam only), --final-rtol/--final-atol, --warm-start, --bubble-source", comm);
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
