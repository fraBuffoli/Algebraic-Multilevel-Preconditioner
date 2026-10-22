/**
 * @file test_factorization.cpp
 * @brief Tests of the symmetric factorizations and of the single precision (P2).
 *
 *  1. Every backend x request (lu, cholesky, ldlt, auto) x precision (double, single) on
 *     - an SPD matrix (bubbles: Cholesky must be used),
 *     - a symmetric indefinite matrix (Cholesky must fail and the chain move on),
 *     - a nonsymmetric matrix (LU whatever the request: a symmetric factorization would be wrong),
 *     - a PSD matrix + sigma I with sigma = 1e-13 ||A|| (as A~_ii of a floating subdomain);
 *     checks: the factorization type chosen, the relative backward error
 *     ||A x - b|| / (||A|| ||x|| + ||b||) (double < 1e-12; single < 1e-5, 1e-3 for the indefinite
 *     matrix whose float LDL^T / LU pivots are small), the numerical
 *     refactorization with new values, transposed solves.
 *  2. Coarse solvers (root, cpardiso, mumps) x request (auto, ldlt, lu) on a distributed SPD
 *     matrix, then update() with symmetric indefinite values (Cholesky must give way to LDL^T
 *     inside the refactorization), then setup() directly on the indefinite matrix: type chosen
 *     and backward error of the solve. With Open MPI cpardiso always uses mtype 11 (see
 *     coarse_solver.cpp); with Intel MPI (G100) this checks the distributed Cholesky.
 *  3. Distributed: the two-level preconditioner with `--local-factorization auto` (Cholesky for
 *     A_ii, A~_ii) and `--coarse-factorization auto` (Cholesky for the root A_00) equals the LU
 *     one; a nonsymmetric problem keeps LU with any request; with `--local-precision single`
 *     GMRES reaches the tolerance on the *true* residual; a time loop with Cholesky + single
 *     precision and setup reuse converges.
 *
 * Run e.g. `mpirun -np 4 ./bin/test_factorization` (a few seconds).
 */
#include "coarse_solver.hpp"
#include "config.hpp"
#include "distributed_matrix.hpp"
#include "domain_decomposer.hpp"
#include "gmres.hpp"
#include "local_solver.hpp"
#include "logger.hpp"
#include "matrix_generators.hpp"
#include "subdomain.hpp"
#include "test_utils.hpp"
#include "time_loop.hpp"
#include "two_level_schwarz.hpp"

#include <algorithm>

using namespace schwarz2lvl;
using linalg::Factorization;

namespace {

core::SpMat generated(const std::string& gen, int nx, double time = 0.0)
{
    utils::GeneratorOptions o;
    o.nx = o.ny = o.nz = nx;
    o.nu = 0.01;
    o.nbubbles = 6;
    o.radius = 0.12;
    o.pulse = 0.5;
    o.pulse_period = 10;
    o.time = time;
    return test::toEigen(utils::MatrixGenerator::generate(gen, o).A);
}

/// @brief A - c * mean(diag(A)) I: symmetric indefinite for c in (0, 1) on a Laplacian.
core::SpMat shifted(const core::SpMat& A, double c)
{
    core::SpMat S = A;
    const double d = A.diagonal().mean();
    for (core::LocalIndex i = 0; i < S.rows(); ++i) S.coeffRef(i, i) -= c * d;
    return S;
}

/// @brief Zero row sums (pure Neumann Laplacian: PSD, singular) + sigma I, sigma = 1e-13 ||A||.
core::SpMat psdPlusShift(const core::SpMat& A)
{
    core::SpMat N = A;
    core::Vec off = core::Vec::Zero(N.rows());
    for (Eigen::Index k = 0; k < N.outerSize(); ++k)
        for (core::SpMat::InnerIterator it(N, k); it; ++it)
            if (it.row() != it.col()) off[it.row()] += it.value();
    const double nrm = 2.0 * (-off).maxCoeff();
    for (core::LocalIndex i = 0; i < N.rows(); ++i) N.coeffRef(i, i) = -off[i] + 1e-13 * nrm;
    return N;
}

double normInf(const core::SpMat& A)
{
    core::Vec rs = core::Vec::Zero(A.rows());
    for (Eigen::Index k = 0; k < A.outerSize(); ++k)
        for (core::SpMat::InnerIterator it(A, k); it; ++it) rs[it.row()] += std::abs(it.value());
    return rs.maxCoeff();
}

double backwardError(const core::SpMat& A, const core::Vec& x, const core::Vec& b)
{
    return (A * x - b).lpNorm<Eigen::Infinity>() / (normInf(A) * x.lpNorm<Eigen::Infinity>() + b.lpNorm<Eigen::Infinity>());
}

} // namespace

// --------------------------------------------------------------------------- 1. local solvers
/**
 * @param kind     Label of the matrix.
 * @param A0, A1   Two matrices with the same pattern (factorize A0, refactorize A1).
 * @param symmetric Whether the matrix is declared symmetric.
 * @param expect   Expected outcome for the symmetric requests: 0 = Cholesky, 1 = not Cholesky, 2 = LU.
 * @param with_single Also test single precision (not for A~_ii-like matrices: sigma is below float precision,
 *                    which is why the eigensolver always factorizes in double).
 */
void checkLocal(const std::string& kind, const core::SpMat& A0, const core::SpMat& A1, bool symmetric, int expect,
                bool with_single = true)
{
    const core::Vec b = core::Vec::LinSpaced(A0.rows(), -1.0, 2.0).array().sin();
    for (const std::string& be : linalg::availableLocalSolvers())
        for (const std::string req : {"lu", "cholesky", "ldlt", "auto"})
            for (const bool single : {false, true}) {
                if (single && !with_single) continue;
                linalg::LocalSolverSpec s;
                s.backend = be;
                s.factorization = req;
                s.symmetric = symmetric;
                s.single = single;
                auto S = linalg::makeLocalSolver(s);
                const bool ok0 = S->factorize(A0);
                core::Vec x0, x1, xt;
                if (ok0) S->solve(b, x0);
                const bool ok1 = ok0 && S->refactorize(A1);
                if (ok1) S->solve(b, x1);
                // Single precision: backward error ~ eps_float x growth. Small for SPD matrices (the use
                // case: A_ii); indefinite LDL^T / LU in float have a larger growth (pivots near zero).
                const double tol = (single && be != "umfpack") ? (expect == 1 ? 1e-3 : 1e-5) : 1e-12;
                const double e0 = ok0 ? backwardError(A0, x0, b) : 1.0, e1 = ok1 ? backwardError(A1, x1, b) : 1.0;
                // Which factorization is acceptable.
                const Factorization f = S->factorization();
                bool type_ok = true;
                if (!symmetric || req == "lu" || be == "umfpack") type_ok = (f == Factorization::LU);
                else if (expect == 0 && (req == "auto" || req == "cholesky")) type_ok = (f == Factorization::Cholesky);
                else if (expect == 0 && req == "ldlt") type_ok = (f == Factorization::LDLT);
                else if (expect == 1) type_ok = (f != Factorization::Cholesky);
                bool trans_ok = true;
                if (ok1 && !symmetric && S->hasTransposeSolve()) {
                    S->solveTranspose(b, xt);
                    trans_ok = backwardError(core::SpMat(A1.transpose()), xt, b) < tol;
                }
                test::check(ok0 && ok1 && e0 < tol && e1 < tol && type_ok && trans_ok,
                            kind + ": " + be + " request " + req + (single ? " single" : "") + " -> " + S->name() +
                                " (backward error " + test::sci(std::max(e0, e1)) + ")");
            }
}

// --------------------------------------------------------------------------- 2. distributed
core::Vec applyOn(const prec::Preconditioner& M, const part::Subdomain& sd)
{
    const core::GlobalIndex n = sd.system().n_global, b = sd.system().rowBegin();
    const core::Vec rg = core::Vec::LinSpaced(static_cast<Eigen::Index>(n), 0.0, 9.0).array().cos();
    core::Vec z;
    M.apply(rg.segment(b, sd.nOwned()), z);
    return z;
}

double grel(const core::Vec& a, const core::Vec& b, MPI_Comm comm)
{
    return std::sqrt(core::allreduceSum((a - b).squaredNorm(), comm) / core::allreduceSum(b.squaredNorm(), comm));
}

/// @brief Expected substring of the coarse solver name for a request (symmetric matrix).
std::string expectedCoarse(const std::string& solver, const std::string& req, bool spd)
{
    if (solver == "mumps") return "";
    if (solver == "cpardiso") {
#ifdef OPEN_MPI
        (void)req;
        (void)spd;
        return "mtype 11 lu";
#else
        if (req == "lu") return "mtype 11 lu";
        if (req == "ldlt") return "mtype -2 ldlt";
        return spd ? "mtype 2 cholesky" : "mtype -2 ldlt (fallback from cholesky)";
#endif
    }
    if (req == "lu") return " lu";
    if (req == "ldlt") return " ldlt";
    return spd ? " cholesky" : " ldlt (fallback from cholesky)";
}

/// @brief Coarse solvers on a distributed matrix: setup (SPD), update (indefinite), setup (indefinite).
void checkCoarse()
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = core::mpiRank(comm), np = core::mpiSize(comm);
    const core::SpMat spd = generated("laplace2d", 14), ind = shifted(spd, 0.37);
    const core::GlobalIndex n = spd.rows();
    std::vector<core::GlobalIndex> offsets(static_cast<std::size_t>(np) + 1);
    for (int r = 0; r <= np; ++r) offsets[static_cast<std::size_t>(r)] = n * r / np;
    const core::GlobalIndex r0 = offsets[static_cast<std::size_t>(rank)], r1 = offsets[static_cast<std::size_t>(rank) + 1];
    // Contributions: every rank gives the entries of its rows (same sequence for setup and update).
    auto entries = [&](const core::SpMat& A) {
        std::vector<core::GlobalTriplet> e;
        for (Eigen::Index j = 0; j < A.outerSize(); ++j)
            for (core::SpMat::InnerIterator it(A, j); it; ++it)
                if (it.row() >= r0 && it.row() < r1) e.push_back({it.row(), it.col(), it.value()});
        return e;
    };
    // Backward error ||A y - c|| / (||A||_inf ||y|| + ||c||) of the distributed solve.
    auto backward = [&](const prec::CoarseSolver& S, const core::SpMat& A) {
        const core::Vec cg = core::Vec::LinSpaced(static_cast<Eigen::Index>(n), 1.0, 5.0).array().sin();
        core::Vec y;
        S.solve(cg.segment(r0, r1 - r0), y);
        std::vector<int> cnt(static_cast<std::size_t>(np)), dsp(static_cast<std::size_t>(np));
        for (int r = 0; r < np; ++r) {
            cnt[static_cast<std::size_t>(r)] = static_cast<int>(offsets[static_cast<std::size_t>(r) + 1] - offsets[static_cast<std::size_t>(r)]);
            dsp[static_cast<std::size_t>(r)] = static_cast<int>(offsets[static_cast<std::size_t>(r)]);
        }
        core::Vec yg(n);
        MPI_Allgatherv(y.data(), static_cast<int>(y.size()), MPI_DOUBLE, yg.data(), cnt.data(), dsp.data(), MPI_DOUBLE, comm);
        double anorm = 0.0;
        for (Eigen::Index i = 0; i < A.rows(); ++i) anorm = std::max(anorm, A.row(i).cwiseAbs().sum());
        return (A * yg - cg).norm() / (anorm * yg.norm() + cg.norm());
    };
    for (const auto& solver : prec::availableCoarseSolvers()) {
        for (const std::string req : {"auto", "ldlt", "lu"}) {
            const std::string what = "coarse " + solver + " request " + req + ": ";
            auto S = prec::makeCoarseSolver(solver, std::max(1, np / 2), "auto", req);
            auto e0 = entries(spd);
            S->setup(comm, offsets, e0, true);
            const std::string n0 = S->name(), x0 = expectedCoarse(solver, req, true);
            const double b0 = backward(*S, spd);
            test::check(n0.find(x0) != std::string::npos && b0 < 1e-12,
                        what + "SPD -> " + n0 + " (backward error " + test::sci(b0) + ")", comm);
            auto e1 = entries(ind);
            S->update(e1);
            const std::string n1 = S->name(), x1 = expectedCoarse(solver, req, false);
            const double b1 = backward(*S, ind);
            test::check(n1.find(x1) != std::string::npos && b1 < 1e-12,
                        what + "update to indefinite -> " + n1 + " (backward error " + test::sci(b1) + ")", comm);
            auto T = prec::makeCoarseSolver(solver, std::max(1, np / 2), "auto", req);
            auto e2 = entries(ind);
            T->setup(comm, offsets, e2, true);
            const std::string n2 = T->name();
            const double b2 = backward(*T, ind);
            test::check(n2.find(x1) != std::string::npos && b2 < 1e-12,
                        what + "setup on indefinite -> " + n2 + " (backward error " + test::sci(b2) + ")", comm);
        }
    }
}

void checkDistributed(const std::string& gen, int nx)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    core::SolverConfig cfg;
    cfg.generate = gen;
    cfg.nx = cfg.ny = cfg.nz = nx;
    cfg.nu = 0.01;
    cfg.nbubbles = 8;
    cfg.radius = 0.1;
    cfg.tau = 0.3;
    cfg.nev = 12;
    cfg.dense_threshold = 0;
    cfg.eig_tol = 1e-12;
    cfg.coarse_solver = "root";
    part::DecompositionInfo info;
    const core::LocalSystem sys = part::DomainDecomposer::decompose(cfg, comm, info);
    part::Subdomain sd(sys, cfg.pou);
    linalg::DistributedMatrix A(sd);
    const std::string label = gen + " " + std::to_string(nx) + (sys.symmetric ? " (symmetric)" : " (nonsymmetric)");

    cfg.local_factorization = cfg.coarse_factorization = "lu";
    prec::TwoLevelSchwarz Mlu(sd, A, cfg);
    cfg.local_factorization = cfg.coarse_factorization = sys.symmetric ? "auto" : "cholesky"; // nonsymmetric: ignored
    prec::TwoLevelSchwarz Mch(sd, A, cfg);
    const auto& st = Mch.stats();
    const bool names = sys.symmetric
                           ? (st.local_factorization.find("cholesky") != std::string::npos &&
                              (Mch.coarse().dimension() == 0 || Mch.coarse().solver().name().find("cholesky") != std::string::npos))
                           : (st.local_factorization.find(" lu") != std::string::npos &&
                              st.atilde_factorization.find(" lu") != std::string::npos);
    const double d = grel(applyOn(Mch, sd), applyOn(Mlu, sd), comm);
    test::check(names && Mch.coarse().dimension() == Mlu.coarse().dimension() && d < 1e-9,
                label + ": request " + cfg.local_factorization + " -> A_ii " + st.local_factorization + ", A~_ii " +
                    st.atilde_factorization + "; same preconditioner as LU (rel. diff " + test::sci(d) + ")",
                comm);

    const core::Vec bb = Eigen::Map<const core::Vec>(sys.rhs.data(), sd.nOwned());
    core::Vec x = core::Vec::Zero(sd.nOwned());
    const solv::GmresResult rd = solv::Gmres(30, 1e-8, 500).solve(A, Mch, bb, x);
    cfg.local_precision = "single";
    prec::TwoLevelSchwarz Msp(sd, A, cfg);
    x.setZero();
    const solv::GmresResult rs = solv::Gmres(30, 1e-8, 500).solve(A, Msp, bb, x);
    test::check(rd.converged && rs.converged && rs.true_rel_residual <= 1.1e-8 &&
                    Msp.stats().local_factorization.find("single") != std::string::npos,
                label + ": single-precision A_ii (" + Msp.stats().local_factorization + "): GMRES " +
                    std::to_string(rs.iterations) + " it. (double " + std::to_string(rd.iterations) +
                    "), true residual " + test::sci(rs.true_rel_residual),
                comm);
}

void checkTimeLoop()
{
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
    cfg.reuse = "refactor";
    cfg.local_factorization = "auto";
    cfg.local_precision = "single";
    const auto r = solv::runTimeLoop(cfg, MPI_COMM_WORLD, utils::Log::nullStream());
    std::string lv;
    for (const auto& s : r) lv += std::to_string(static_cast<int>(s.level));
    const bool conv = std::all_of(r.begin(), r.end(), [](const solv::SolveRecord& s) { return s.converged && s.true_rel_residual <= 1.1e-8; });
    test::check(lv == "201010" && conv, "time loop, Cholesky + single precision + refactorizations: levels " + lv + ", all converged");
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    try {
        const core::SpMat spd0 = generated("bubbles2d", 24), spd1 = generated("bubbles2d", 24, 3.0);
        checkLocal("SPD (bubbles2d)", spd0, spd1, true, 0);
        const core::SpMat spd3 = generated("bubbles3d", 10), spd4 = generated("bubbles3d", 10, 3.0);
        checkLocal("SPD (bubbles3d)", spd3, spd4, true, 0);
        const core::SpMat lap = generated("laplace2d", 16);
        checkLocal("symmetric indefinite (shifted Laplacian)", shifted(lap, 0.37), shifted(lap, 0.41), true, 1);
        checkLocal("nonsymmetric (convdiff2d)", generated("convdiff2d", 16), generated("convdiff2d", 16, 4.0), false, 2);
        checkLocal("PSD + 1e-13 ||A|| I (floating subdomain)", psdPlusShift(spd0), psdPlusShift(spd1), true, 0, false);
        checkCoarse();
        checkDistributed("bubbles2d", 32);
        checkDistributed("laplace2d", 20);
        checkDistributed("convdiff2d", 24);
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
