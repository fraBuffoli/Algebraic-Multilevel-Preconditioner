/**
 * @file test_reuse.cpp
 * @brief Tests of the setup reuse between solves with the same sparsity pattern (P1).
 *
 *  1. ReusePolicy: decisions for given iteration counts (pure logic).
 *  2. LocalSolver::refactorize(): same solution as a fresh factorize() on new values, every backend.
 *  3. On time-dependent problems (bubbles2d: SPD; convdiff2d / coupled2d with pulsating convection:
 *     nonsymmetric), for both partitions of unity and every coarse solver of the build:
 *     a. unchanged matrix: Subdomain::updateValues() reports no change, update() -> Reuse,
 *        bitwise identical application;
 *     b. DomainDecomposer::updateValues(): the rows of every rank are those of the generated
 *        matrix of the new step, permuted;
 *     c. Subdomain::updateValues(): A_ii, A(Omega_I, Omega_i), lumping sums identical to a
 *        Subdomain built from scratch on the new values;
 *     d. update(Refactorize): the coarse correction is consistent with the *new* A,
 *        R_0 (r - A R_0^T A_00^{-1} R_0 r) = 0 (and it is not before the update), and the
 *        application equals a sequential reference built from the new global matrix and the
 *        (kept) coarse basis; GMRES converges;
 *     e. update(Rebuild): identical to a preconditioner constructed from scratch.
 *  4. runTimeLoop(): levels of the auto and off modes, convergence, and a sequence read
 *     from Matrix Market files (--matrix with a %d pattern) giving the same iterations as
 *     the generator.
 *
 * Run e.g. `mpirun -np 4 ./bin/test_reuse` (a few seconds).
 */
#include "config.hpp"
#include "distributed_matrix.hpp"
#include "domain_decomposer.hpp"
#include "gmres.hpp"
#include "local_solver.hpp"
#include "logger.hpp"
#include "matrix_generators.hpp"
#include "matrix_market_io.hpp"
#include "reuse_policy.hpp"
#include "subdomain.hpp"
#include "test_utils.hpp"
#include "time_loop.hpp"
#include "two_level_schwarz.hpp"
#include "coarse_solver.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cstdio>

using namespace schwarz2lvl;
using prec::UpdateLevel;

// --------------------------------------------------------------------------- 1. policy
void checkPolicy()
{
    using P = prec::ReusePolicy;
    bool ok = std::abs(P::perDecade(10, 1e-5) - 2.0) < 1e-12 && P::perDecade(0, 1e-5) == 0.0 && P::perDecade(5, 1.0) == 0.0;
    test::check(ok, "policy: iterations per decade");

    // Auto, break-even rule: rebuild costs 2.0 s, refactorize 0.2 s -> premium 1.8 s.
    // Each solve reduces the residual by 1e-8; the reference needs 10 iterations (q_ref = 1.25)
    // and an iteration costs 0.1 s.
    P a(P::Mode::Auto, 0.0, 0);
    a.record(UpdateLevel::Rebuild, 10, 1e-8, true, 2.0, 1.0);         // reference, nothing wasted
    ok = a.decide(false) == UpdateLevel::Reuse && a.decide(true) == UpdateLevel::Refactorize && a.wasted() == 0.0;
    a.record(UpdateLevel::Refactorize, 15, 1e-8, true, 0.2, 1.5);     // 5 extra iterations: 0.5 s wasted
    ok = ok && std::abs(a.wasted() - 0.5) < 1e-12 && a.decide(true) == UpdateLevel::Refactorize;
    a.record(UpdateLevel::Reuse, 15, 1e-8, true, 0.0, 1.5);           // same stale basis: +0.5 s
    a.record(UpdateLevel::Refactorize, 19, 1e-8, true, 0.2, 1.9);     // +0.9 s -> 1.9 s > premium 1.8 s
    ok = ok && std::abs(a.wasted() - 1.9) < 1e-12 && std::abs(a.rebuildPremium() - 1.8) < 1e-12;
    ok = ok && a.decide(true) == UpdateLevel::Rebuild && a.decide(false) == UpdateLevel::Reuse;
    a.record(UpdateLevel::Rebuild, 12, 1e-8, true, 2.5, 1.2);         // new reference 1.5, counter reset
    ok = ok && a.decide(true) == UpdateLevel::Refactorize && a.wasted() == 0.0 && std::abs(a.reference() - 1.5) < 1e-12;
    test::check(ok, "policy auto: break-even rule (rebuild when the time lost reaches the rebuild premium)");

    P g(P::Mode::Auto, 1.5, 0);                                       // safety net on convergence
    g.record(UpdateLevel::Rebuild, 10, 1e-8, true, 1e9, 1.0);         // huge rebuild cost: break-even never reached
    g.record(UpdateLevel::Refactorize, 16, 1e-8, true, 0.1, 1.6);     // q = 2.0 > 1.5 * 1.25
    ok = g.decide(true) == UpdateLevel::Rebuild;
    P n(P::Mode::Auto, 0.0, 0);
    n.record(UpdateLevel::Rebuild, 10, 1e-8, true, 1e9, 1.0);
    n.record(UpdateLevel::Refactorize, 500, 1e-3, false, 0.1, 50.0);  // not converged
    ok = ok && n.decide(true) == UpdateLevel::Rebuild;
    P b(P::Mode::Auto, 0.0, 2);                                       // at most 2 refactorizations
    b.record(UpdateLevel::Rebuild, 10, 1e-8, true, 1e9, 1.0);
    ok = ok && b.decide(true) == UpdateLevel::Refactorize;
    b.record(UpdateLevel::Refactorize, 10, 1e-8, true, 0.1, 1.0);
    ok = ok && b.decide(true) == UpdateLevel::Refactorize;
    b.record(UpdateLevel::Refactorize, 10, 1e-8, true, 0.1, 1.0);
    ok = ok && b.decide(true) == UpdateLevel::Rebuild;
    test::check(ok, "policy auto: safety nets (--rebuild-growth, --rebuild-every, non-convergence)");

    P c(P::Mode::Off, 0.0, 0);
    test::check(c.decide(false) == UpdateLevel::Rebuild && c.decide(true) == UpdateLevel::Rebuild,
                "policy off: rebuild before every solve");
    P r(P::Mode::Refactor, 0.0, 0);
    r.record(UpdateLevel::Rebuild, 10, 1e-8, true, 2.0, 1.0);
    r.record(UpdateLevel::Refactorize, 900, 1e-3, false, 0.1, 90.0);
    test::check(r.decide(false) == UpdateLevel::Reuse && r.decide(true) == UpdateLevel::Refactorize,
                "policy refactor: never rebuild");
}

// --------------------------------------------------------------------------- 2. local solvers
void checkLocalRefactorize(const std::string& gen, int nx)
{
    utils::GeneratorOptions o;
    o.nx = o.ny = o.nz = nx;
    o.nu = 0.01;
    o.ncomp = 2;
    o.pulse = 0.5;
    o.pulse_period = 10;
    o.nbubbles = 6;
    o.radius = 0.12;
    const core::SpMat A0 = test::toEigen(utils::MatrixGenerator::generate(gen, o).A);
    o.time = 4.0;
    const core::SpMat A1 = test::toEigen(utils::MatrixGenerator::generate(gen, o).A);
    const core::Vec b = core::Vec::LinSpaced(A1.rows(), -1.0, 2.0).array().sin();
    for (const std::string& name : linalg::availableLocalSolvers()) {
        auto s = linalg::makeLocalSolver(name), f = linalg::makeLocalSolver(name);
        bool ok = s->factorize(A0) && s->refactorize(A1) && f->factorize(A1);
        core::Vec xs, xf;
        if (ok) {
            s->solve(b, xs);
            f->solve(b, xf);
        }
        const double diff = ok ? test::relDiff(xs, xf) : 1.0;
        const double res = ok ? (A1 * xs - b).norm() / b.norm() : 1.0;
        test::check(ok && diff < 1e-10 && res < 1e-10,
                    gen + ": " + name + " refactorize vs factorize (diff " + test::sci(diff) + ", residual " +
                        test::sci(res) + (s->refactorizationFellBack() ? ", fell back" : "") + ")");
    }
}

// --------------------------------------------------------------------------- 3. distributed
/// @brief Global 2-norm of distributed vectors (collective).
double gnorm(const core::Vec& v, MPI_Comm comm) { return std::sqrt(core::allreduceSum(v.squaredNorm(), comm)); }

/// @brief Relative difference of two distributed vectors (collective).
double grel(const core::Vec& a, const core::Vec& b, MPI_Comm comm) { return gnorm(a - b, comm) / gnorm(b, comm); }

/**
 * @brief Application of M against a sequential reference on rank 0 built from the global matrix @p Ag
 *        and the coarse basis gathered from M (as in test_preconditioner). Returns the relative error.
 */
double referenceError(const part::Subdomain& sd, const prec::TwoLevelSchwarz& M, const core::SolverConfig& cfg,
                      const part::CsrMatrix& Ag, const core::Vec& rg)
{
    MPI_Comm comm = sd.comm();
    const int rank = core::mpiRank(comm), np = core::mpiSize(comm);
    const core::GlobalIndex n = sd.system().n_global, b = sd.system().rowBegin();
    core::Vec z;
    M.apply(rg.segment(b, sd.nOwned()), z);
    const std::vector<core::Scalar> zg = part::DomainDecomposer::gather(sd.system(), z);

    const int nl = sd.nLocal(), m = static_cast<int>(M.coarse().basis().cols());
    std::vector<core::GlobalIndex> ids(static_cast<std::size_t>(nl));
    for (core::LocalIndex l = 0; l < nl; ++l) ids[static_cast<std::size_t>(l)] = sd.globalId(l);
    std::vector<int> cn(static_cast<std::size_t>(np)), cm(static_cast<std::size_t>(np)), cv(static_cast<std::size_t>(np));
    MPI_Gather(&nl, 1, MPI_INT, cn.data(), 1, MPI_INT, 0, comm);
    MPI_Gather(&m, 1, MPI_INT, cm.data(), 1, MPI_INT, 0, comm);
    const int nv = nl * m;
    MPI_Gather(&nv, 1, MPI_INT, cv.data(), 1, MPI_INT, 0, comm);
    const std::vector<int> dn = core::displacements(cn), dv = core::displacements(cv);
    std::vector<core::GlobalIndex> all_ids(rank == 0 ? static_cast<std::size_t>(dn.back() + cn.back()) : 0);
    std::vector<core::Scalar> all_D(all_ids.size()), all_V(rank == 0 ? static_cast<std::size_t>(dv.back() + cv.back()) : 0);
    MPI_Gatherv(ids.data(), nl, core::mpiType<core::GlobalIndex>(), all_ids.data(), cn.data(), dn.data(),
                core::mpiType<core::GlobalIndex>(), 0, comm);
    MPI_Gatherv(sd.pou().data(), nl, MPI_DOUBLE, all_D.data(), cn.data(), dn.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(M.coarse().basis().data(), nv, MPI_DOUBLE, all_V.data(), cv.data(), dv.data(), MPI_DOUBLE, 0, comm);

    double err = 0.0;
    if (rank == 0) {
        const core::Mat Ad = core::Mat(test::toEigen(Ag));
        const auto nn = static_cast<Eigen::Index>(n);
        const core::GlobalIndex n0 = M.coarse().dimension();
        core::Mat R0T = core::Mat::Zero(nn, static_cast<Eigen::Index>(n0));
        auto oneLevel = [&](const core::Vec& r) {
            core::Vec out = core::Vec::Zero(nn);
            for (int p = 0; p < np; ++p) {
                const int ni = cn[static_cast<std::size_t>(p)];
                const core::GlobalIndex* id = all_ids.data() + dn[static_cast<std::size_t>(p)];
                core::Mat Aii(ni, ni);
                core::Vec ri(ni);
                for (int a = 0; a < ni; ++a) {
                    ri[a] = r[id[a]];
                    for (int c = 0; c < ni; ++c) Aii(a, c) = Ad(id[a], id[c]);
                }
                const core::Vec yi = Aii.partialPivLu().solve(ri);
                for (int a = 0; a < ni; ++a)
                    out[id[a]] += ((cfg.one_level == "ras") ? all_D[static_cast<std::size_t>(dn[static_cast<std::size_t>(p)] + a)] : 1.0) * yi[a];
            }
            return out;
        };
        Eigen::Index c0 = 0;
        for (int p = 0; p < np; ++p) {
            const int ni = cn[static_cast<std::size_t>(p)], mi = cm[static_cast<std::size_t>(p)];
            const core::GlobalIndex* id = all_ids.data() + dn[static_cast<std::size_t>(p)];
            const Eigen::Map<const core::Mat> Vi(all_V.data() + dv[static_cast<std::size_t>(p)], ni, mi);
            for (int a = 0; a < ni; ++a)
                for (int c = 0; c < mi; ++c) R0T(id[a], c0 + c) += Vi(a, c);
            c0 += mi;
        }
        const core::Mat E = R0T.transpose() * Ad * R0T; // A_00 of the global matrix Ag
        const Eigen::PartialPivLU<core::Mat> Elu(E);
        auto Q = [&](const core::Vec& r) { return core::Vec(R0T * Elu.solve(R0T.transpose() * r)); };
        core::Vec zref;
        if (n0 == 0) zref = oneLevel(rg);
        else if (cfg.coarse == "additive") zref = Q(rg) + oneLevel(rg);
        else {
            const core::Vec x0 = Q(rg);
            zref = x0 + oneLevel(rg - Ad * x0);
        }
        err = test::relDiff(Eigen::Map<const core::Vec>(zg.data(), nn), zref);
    }
    MPI_Bcast(&err, 1, MPI_DOUBLE, 0, comm);
    return err;
}

/// @brief ||R_0 (r - A Q r)|| / ||R_0 r||, Q = R_0^T A_00^{-1} R_0: zero iff A_00 = R_0 A R_0^T for the current A.
double coarseConsistency(const prec::TwoLevelSchwarz& M, const linalg::DistributedMatrix& A, const core::Vec& r)
{
    MPI_Comm comm = A.comm();
    if (M.coarse().dimension() == 0) return 0.0; // empty coarse space (e.g. a single subdomain)
    core::Vec x0, w, c, cr;
    M.coarse().apply(r, x0);
    A.apply(x0, w);
    w = r - w;
    M.coarse().restrict(w, c);
    M.coarse().restrict(r, cr);
    return gnorm(c, comm) / gnorm(cr, comm);
}

/// @brief Rows of @p sys == rows of the global permuted matrix (rank 0's copy broadcast).
bool rowsMatch(const core::LocalSystem& sys, part::CsrMatrix Ag)
{
    test::bcastCsr(Ag, sys.comm);
    const core::GlobalIndex b = sys.rowBegin();
    for (core::LocalIndex l = 0; l < sys.nOwned(); ++l) {
        const auto g = static_cast<std::size_t>(b + l);
        const auto k0 = Ag.row_ptr[g], k1 = Ag.row_ptr[g + 1];
        if (k1 - k0 != sys.row_ptr[static_cast<std::size_t>(l) + 1] - sys.row_ptr[static_cast<std::size_t>(l)]) return false;
        for (auto k = k0, q = sys.row_ptr[static_cast<std::size_t>(l)]; k < k1; ++k, ++q)
            if (Ag.col[static_cast<std::size_t>(k)] != sys.col[static_cast<std::size_t>(q)] ||
                Ag.val[static_cast<std::size_t>(k)] != sys.val[static_cast<std::size_t>(q)])
                return false;
    }
    return true;
}

/// @brief A_perm(i, j) == generated matrix of the step at (new_to_old[i], new_to_old[j]) (rank 0).
bool permutedMatchesGenerated(const core::SolverConfig& cfg, int step, const part::DecompositionInfo& info, MPI_Comm comm)
{
    int ok = 1;
    if (core::mpiRank(comm) == 0) {
        utils::GeneratorOptions o;
        o.nx = cfg.nx;
        o.ny = cfg.ny;
        o.nz = cfg.nz;
        o.nu = cfg.nu;
        o.ncomp = cfg.ncomp;
        o.time = step;
        o.pulse = cfg.pulse;
        o.pulse_period = cfg.pulse_period;
        o.rho_ratio = cfg.rho_ratio;
        o.nbubbles = cfg.nbubbles;
        o.radius = cfg.radius;
        o.cfl = cfg.cfl;
        const core::SpMat G = test::toEigen(utils::MatrixGenerator::generate(cfg.generate, o).A);
        const part::CsrMatrix& P = info.A_perm;
        for (core::GlobalIndex i = 0; i < P.nrows && ok; ++i)
            for (auto k = P.row_ptr[static_cast<std::size_t>(i)]; k < P.row_ptr[static_cast<std::size_t>(i) + 1]; ++k)
                if (G.coeff(static_cast<core::LocalIndex>(info.new_to_old[static_cast<std::size_t>(i)]),
                            static_cast<core::LocalIndex>(info.new_to_old[static_cast<std::size_t>(P.col[static_cast<std::size_t>(k)])])) !=
                    P.val[static_cast<std::size_t>(k)]) {
                    ok = 0;
                    break;
                }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
    return ok != 0;
}

void runCase(core::SolverConfig cfg, const std::string& pou, const std::string& coarse_solver, int coarse_procs, bool full)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    cfg.pou = pou;
    cfg.coarse_solver = coarse_solver;
    cfg.coarse_procs = coarse_procs;
    const std::string label = cfg.generate + "/" + pou + "/" + coarse_solver + (coarse_procs ? ":" + std::to_string(coarse_procs) : "");
    part::DecompositionInfo info;
    info.keep_global = true;
    core::LocalSystem sys = part::DomainDecomposer::decompose(cfg, comm, info);
    part::Subdomain sd(sys, pou);
    linalg::DistributedMatrix A(sd);
    prec::TwoLevelSchwarz M(sd, A, cfg);
    const core::GlobalIndex n = sys.n_global, b = sys.rowBegin();
    const core::Vec rg = core::Vec::LinSpaced(static_cast<Eigen::Index>(n), 0.0, 9.0).array().cos();
    const core::Vec r = rg.segment(b, sd.nOwned());

    // (a) unchanged matrix
    core::Vec z0, z1;
    M.apply(r, z0);
    const bool changed0 = sd.updateValues();
    const UpdateLevel l0 = M.update(UpdateLevel::Refactorize);
    M.apply(r, z1);
    test::check(!changed0 && l0 == UpdateLevel::Reuse && (z1 - z0).norm() == 0.0 && sd.valuesVersion() == 0,
                label + ": unchanged matrix -> reuse, identical application", comm);

    // (b) new values of step 3
    const int step = 3;
    part::DomainDecomposer::updateValues(cfg, step, sys, info);
    if (full) {
        test::check(permutedMatchesGenerated(cfg, step, info, comm), label + ": updateValues = generated matrix of the step, permuted", comm);
        test::check(rowsMatch(sys, info.A_perm), label + ": updateValues sends every rank its rows", comm);
    }

    // (c) subdomain values vs a subdomain built from scratch
    const bool changed1 = sd.updateValues();
    {
        part::Subdomain fresh(sys, pou);
        const bool same = (core::SpMat(sd.Aii() - fresh.Aii())).norm() == 0.0 &&
                          (core::SpMatRow(sd.Aowned() - fresh.Aowned())).norm() == 0.0 &&
                          (sd.lumpingSums() - fresh.lumpingSums()).norm() == 0.0;
        test::check(changed1 && sd.valuesVersion() == 1 && same && !M.upToDate(),
                    label + ": Subdomain::updateValues = fresh subdomain (A_ii, A(Omega_I,Omega_i), lumping sums)", comm);
    }

    // (d) refactorize: coarse operator of the new A with the old basis
    const double stale = coarseConsistency(M, A, r); // A_00 still of the old matrix
    const core::GlobalIndex n0 = M.coarse().dimension();
    const UpdateLevel l1 = M.update(UpdateLevel::Refactorize);
    const double cons = coarseConsistency(M, A, r);
    test::check(l1 == UpdateLevel::Refactorize && M.upToDate() && M.coarse().dimension() == n0 && cons < 1e-10 &&
                    (stale > 1e-6 || n0 == 0),
                label + ": refactorize -> R_0 (r - A Q r) = 0 for the new A (" + test::sci(cons) + ", before " +
                    test::sci(stale) + ")",
                comm);
    const double err = referenceError(sd, M, cfg, info.A_perm, rg);
    test::check(err < 1e-9, label + ": refactorize -> apply vs reference with the new matrix (rel. err " + test::sci(err) + ")", comm);
    {
        const core::Vec bb = Eigen::Map<const core::Vec>(sys.rhs.data(), sd.nOwned());
        core::Vec x = core::Vec::Zero(sd.nOwned());
        const solv::GmresResult res = solv::Gmres(30, 1e-8, 500).solve(A, M, bb, x);
        test::check(res.converged && res.true_rel_residual < 1e-7,
                    label + ": refactorize -> GMRES converged in " + std::to_string(res.iterations) + " it.", comm);
    }
    // A Refactorize request on an up-to-date preconditioner does nothing.
    test::check(M.update(UpdateLevel::Refactorize) == UpdateLevel::Reuse, label + ": second refactorize request -> reuse", comm);

    // (e) rebuild = constructed from scratch
    if (full) {
        const UpdateLevel l2 = M.update(UpdateLevel::Rebuild);
        prec::TwoLevelSchwarz F(sd, A, cfg);
        core::Vec zm, zf;
        M.apply(r, zm);
        F.apply(r, zf);
        const double d = grel(zm, zf, comm);
        test::check(l2 == UpdateLevel::Rebuild && M.coarse().dimension() == F.coarse().dimension() && d < 1e-8,
                    label + ": rebuild = new preconditioner (n0 " + std::to_string(F.coarse().dimension()) + ", rel. diff " +
                        test::sci(d) + ")",
                    comm);
    }
}

void runProblem(const std::string& gen, int nx)
{
    core::SolverConfig cfg;
    cfg.generate = gen;
    cfg.nx = cfg.ny = cfg.nz = nx;
    cfg.nu = 0.01;
    cfg.ncomp = 2;
    cfg.pulse = 0.5;
    cfg.pulse_period = 10;
    cfg.nbubbles = 8;
    cfg.radius = 0.1;
    cfg.cfl = 1.0;
    cfg.tau = 0.3;
    cfg.nev = 12;
    cfg.dense_threshold = 0;
    cfg.eig_tol = 1e-12;
    if (core::mpiRank(MPI_COMM_WORLD) == 0) std::cout << "case " << gen << " " << nx << ", np=" << core::mpiSize(MPI_COMM_WORLD) << "\n";
    runCase(cfg, "boolean", "root", 0, true);
    runCase(cfg, "multiplicity", "auto", 0, true);
    for (const std::string& cs : prec::availableCoarseSolvers()) {
        if (cs == "root") continue;
        runCase(cfg, "boolean", cs, 0, false);
        runCase(cfg, "boolean", cs, 2, false);
    }
}

// --------------------------------------------------------------------------- 4. time loop
void checkTimeLoop()
{
    MPI_Comm comm = MPI_COMM_WORLD;
    core::SolverConfig cfg;
    cfg.generate = "bubbles2d";
    cfg.nx = cfg.ny = cfg.nz = 40;
    cfg.nbubbles = 6;
    cfg.radius = 0.1;
    cfg.cfl = 1.0;
    cfg.tau = 0.3;
    cfg.nev = 12;
    cfg.steps = 3;
    cfg.correctors = 2;
    cfg.reuse = "refactor"; // levels independent of the timings
    auto levels = [](const std::vector<solv::SolveRecord>& v) {
        std::string s;
        for (const auto& r : v) s += std::to_string(static_cast<int>(r.level));
        return s;
    };
    auto converged = [](const std::vector<solv::SolveRecord>& v) {
        return std::all_of(v.begin(), v.end(), [](const solv::SolveRecord& r) { return r.converged; });
    };
    const auto ra = solv::runTimeLoop(cfg, comm, utils::Log::nullStream());
    test::check(levels(ra) == "201010" && converged(ra), "time loop refactor: levels " + levels(ra) + " (expected 201010), all converged", comm);
    cfg.reuse = "off";
    const auto ro = solv::runTimeLoop(cfg, comm, utils::Log::nullStream());
    test::check(levels(ro) == "222222" && converged(ro), "time loop off: levels " + levels(ro) + " (expected 222222), all converged", comm);
    test::check(ra.front().iterations == ro.front().iterations && ra[1].iterations == ro[1].iterations,
                "time loop: refactor and off identical on the first step", comm);
    cfg.reuse = "auto";
    const auto rb = solv::runTimeLoop(cfg, comm, utils::Log::nullStream());
    bool levels_ok = rb.size() == 6 && converged(rb);
    for (std::size_t k = 0; levels_ok && k < rb.size(); ++k) // correctors reuse; new matrices: refactorize or rebuild
        levels_ok = (k == 0) ? rb[k].level == UpdateLevel::Rebuild
                           : ((k % 2 == 1) ? rb[k].level == UpdateLevel::Reuse : rb[k].level != UpdateLevel::Reuse);
    test::check(levels_ok, "time loop auto: levels " + levels(rb) + ", all converged", comm);

    // Same sequence read from Matrix Market files: --matrix with a %d pattern.
    const std::string pattern = "test_reuse_seq_%02d.mtx";
    if (core::mpiRank(comm) == 0) {
        utils::GeneratorOptions o;
        o.nx = o.ny = cfg.nx;
        o.nbubbles = cfg.nbubbles;
        o.radius = cfg.radius;
        o.cfl = cfg.cfl;
        o.rho_ratio = cfg.rho_ratio;
        for (int s = 0; s < cfg.steps; ++s) {
            o.time = s;
            char name[64];
            std::snprintf(name, sizeof(name), pattern.c_str(), s);
            part::MatrixMarketIO::writeMatrix(name, utils::MatrixGenerator::generate(cfg.generate, o).A);
        }
    }
    MPI_Barrier(comm);
    core::SolverConfig cf = cfg;
    cf.generate.clear();
    cf.matrix_file = pattern;
    cf.reuse = "refactor";
    const auto rf = solv::runTimeLoop(cf, comm, utils::Log::nullStream());
    bool same = rf.size() == ra.size();
    for (std::size_t k = 0; same && k < rf.size(); ++k) same = rf[k].iterations == ra[k].iterations && rf[k].level == ra[k].level;
    test::check(same, "time loop from files (--matrix " + pattern + "): same levels and iterations as the generator", comm);
    if (core::mpiRank(comm) == 0)
        for (int s = 0; s < cfg.steps; ++s) {
            char name[64];
            std::snprintf(name, sizeof(name), pattern.c_str(), s);
            std::remove(name);
        }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    try {
        checkPolicy();
        checkLocalRefactorize("bubbles2d", 24);
        checkLocalRefactorize("convdiff2d", 24);
        checkLocalRefactorize("coupled2d", 12);
        runProblem("bubbles2d", 32);
        runProblem("convdiff2d", 24);
        runProblem("coupled2d", 12);
        checkTimeLoop();
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int f = test::failures();
    if (core::mpiRank(MPI_COMM_WORLD) == 0) std::cout << (f ? "FAILED" : "ALL PASSED") << std::endl;
    MPI_Finalize();
    return f ? 1 : 0;
}
