/**
 * @file test_preconditioner.cpp
 * @brief Validation of the setup and of the preconditioners against sequential references.
 *
 * Checks, for several problems, partitions of unity and process counts:
 *  1. local eigensolver: sparse (Spectra) path vs dense reference path
 *     (kernel dimensions, eigenvalues, selected subspace), kernel residual
 *     ||A~ Q|| and eigen-residual ||Pi_R B u - lambda A~ u|| of (3.1);
 *  2. application of RAS/ASM with deflated/additive coarse correction against
 *     a reference assembled on rank 0 from the global matrix and the gathered
 *     index sets Omega_i, partitions of unity D_i and local bases V_i
 *     (R_0^T = [R_1^T V_1 ... R_N^T V_N], A_00 = R_0 A R_0^T, dense solves);
 *  3. GMRES convergence (true residual).
 *
 * Run e.g. `mpirun -np 4 ./bin/test_preconditioner`.
 */
#include "config.hpp"
#include "distributed_matrix.hpp"
#include "domain_decomposer.hpp"
#include "gmres.hpp"
#include "local_block_splitting.hpp"
#include "local_eigensolver.hpp"
#include "subdomain.hpp"
#include "test_utils.hpp"
#include "two_level_schwarz.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace schwarz2lvl;

/// @brief Largest principal-angle sine between span(X) and span(Y) (same dimension).
double subspaceDistance(const core::Mat& X, const core::Mat& Y)
{
    if (X.cols() == 0 && Y.cols() == 0) return 0.0;
    if (X.cols() != Y.cols()) return 1.0;
    const core::Mat Qx = Eigen::HouseholderQR<core::Mat>(X).householderQ() * core::Mat::Identity(X.rows(), X.cols());
    const core::Mat Qy = Eigen::HouseholderQR<core::Mat>(Y).householderQ() * core::Mat::Identity(Y.rows(), Y.cols());
    return (Qy - Qx * (Qx.transpose() * Qy)).norm();
}

/// @brief Test 1: sparse vs dense local eigensolver on the calling rank's subdomain.
void checkEigensolver(const part::Subdomain& sd, double tau, int nev)
{
    const core::SpMat At = prec::buildLocalBlockSplitting(sd);
    prec::LocalEigensolverOptions o;
    o.tau = tau;
    o.nev = nev;
    o.tol = 1e-12;
    o.dense_threshold = 0;
    o.symmetric = sd.system().symmetric;
    const prec::LocalEigensolver solver(o);
    const prec::LocalEigensolverResult sp = solver.compute(sd.Aii(), At, sd.pou(), false);
    const prec::LocalEigensolverResult de = solver.compute(sd.Aii(), At, sd.pou(), true);

    test::check(sp.kernel_dim == de.kernel_dim && sp.kernel_part == de.kernel_part,
                "kernel dimensions (sparse inverse iteration vs dense SVD)", sd.comm());
    // Kernel part of Z: in ker(A~).
    const double nA = core::Mat(At).norm();
    const double kres = (sp.kernel_part > 0) ? (At * sp.Z.leftCols(sp.kernel_part)).norm() / nA : 0.0;
    test::check(kres < 1e-10, "(L∩K)^⊥_K ⊂ ker(A~_ii)", sd.comm());

    // Every eigenvalue computed by Spectra is an eigenvalue of the dense problem
    // (nearest match: conjugate pairs / equal moduli may be ordered differently).
    double ediff = 0.0;
    for (const auto& ls : sp.eigenvalues) {
        double best = 1e300;
        for (const auto& ld : de.eigenvalues) best = std::min(best, std::abs(ls - ld) / std::abs(ld));
        ediff = std::max(ediff, best);
    }
    ediff = core::allreduceMax(ediff, sd.comm());
    test::check(ediff < 1e-6, "eigenvalues of (3.1): Spectra vs dense (max rel. diff " + test::sci(ediff) + ")",
                sd.comm());

    // Eigen-residual of (3.1) for the selected real eigenvectors: ||Pi_R B u - lambda A~ u||.
    const core::Mat Ad = core::Mat(At);
    const core::Mat B = sd.pou().asDiagonal() * core::Mat(sd.Aii()) * sd.pou().asDiagonal();
    Eigen::JacobiSVD<core::Mat> svd(Ad, Eigen::ComputeFullU);
    Eigen::Index r = 0;
    while (r < svd.singularValues().size() && svd.singularValues()[r] > 1e-8 * svd.singularValues()[0]) ++r;
    const core::Mat P = svd.matrixU().rightCols(Ad.rows() - r);
    double eres = 0.0;
    int col = sp.kernel_part;
    for (const auto& l : sp.eigenvalues) {
        if (std::abs(l) < 1.0 / tau) continue;
        if (col >= sp.Z.cols()) break;
        if (std::abs(l.imag()) > 1e-10 * std::abs(l)) break; // complex pairs: skip the residual test
        const core::Vec u = sp.Z.col(col++);
        core::Vec lhs = B * u;
        lhs -= P * (P.transpose() * lhs);
        eres = std::max(eres, (lhs - l.real() * (Ad * u)).norm() / (std::abs(l) * nA));
    }
    test::check(eres < 1e-6, "eigen-residual ||Pi_R B u - lambda A~ u|| of (3.1)", sd.comm());

    // Same selected subspace.
    const double dist = subspaceDistance(sp.Z, de.Z);
    test::check(sp.Z.cols() == de.Z.cols() && dist < 1e-5, "selected subspace Z_i: Spectra vs dense", sd.comm());
}

/// @brief Test 2: distributed preconditioner vs reference on rank 0.
void checkPreconditioner(const part::Subdomain& sd, const linalg::DistributedMatrix& A, const core::SolverConfig& cfg,
                         const part::CsrMatrix& Ag, const std::string& label)
{
    MPI_Comm comm = sd.comm();
    const int rank = core::mpiRank(comm), np = core::mpiSize(comm);
    prec::TwoLevelSchwarz M(sd, A, cfg);

    // Distributed application.
    const core::GlobalIndex n = sd.system().n_global; // Ag is only available on rank 0
    const core::Vec rg = core::Vec::LinSpaced(static_cast<Eigen::Index>(n), 0.0, 7.0).array().cos();
    const core::GlobalIndex b = sd.system().rowBegin();
    core::Vec z;
    M.apply(rg.segment(b, sd.nOwned()), z);
    const std::vector<core::Scalar> zg = part::DomainDecomposer::gather(sd.system(), z);

    // Gather Omega_i, D_i and V_i on rank 0.
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
    MPI_Gatherv(ids.data(), nl, core::mpiType<core::GlobalIndex>(), all_ids.data(), cn.data(), dn.data(), core::mpiType<core::GlobalIndex>(), 0, comm);
    MPI_Gatherv(sd.pou().data(), nl, MPI_DOUBLE, all_D.data(), cn.data(), dn.data(), MPI_DOUBLE, 0, comm);
    MPI_Gatherv(M.coarse().basis().data(), nv, MPI_DOUBLE, all_V.data(), cv.data(), dv.data(), MPI_DOUBLE, 0, comm);

    bool ok = true;
    double err = 0.0;
    if (rank == 0) {
        const core::Mat Ad = core::Mat(test::toEigen(Ag));
        const Eigen::Index nn = static_cast<Eigen::Index>(n);
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
                core::Vec yi = Aii.partialPivLu().solve(ri);
                for (int a = 0; a < ni; ++a) {
                    const double d = (cfg.one_level == "ras") ? all_D[static_cast<std::size_t>(dn[static_cast<std::size_t>(p)] + a)] : 1.0;
                    out[id[a]] += d * yi[a];
                }
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
        const core::Mat E = R0T.transpose() * Ad * R0T;
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
        ok = err < 1e-9;
    }
    MPI_Bcast(&err, 1, MPI_DOUBLE, 0, comm);
    test::check(ok, label + ": apply vs sequential reference (rel. err " + test::sci(err) + ")", comm);

    // GMRES with this preconditioner.
    const core::Vec bb = Eigen::Map<const core::Vec>(sd.system().rhs.data(), sd.nOwned());
    core::Vec x = core::Vec::Zero(sd.nOwned());
    const solv::GmresResult res = solv::Gmres(30, 1e-8, 500).solve(A, M, bb, x);
    test::check(res.converged && res.true_rel_residual < 1e-7,
                label + ": GMRES converged in " + std::to_string(res.iterations) + " it.", comm);
}

void runCase(const std::string& gen, int nx, int ncomp, double nu)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    core::SolverConfig cfg;
    cfg.generate = gen;
    cfg.nx = cfg.ny = cfg.nz = nx;
    cfg.nu = nu;
    cfg.ncomp = ncomp;
    cfg.tau = 0.3;
    cfg.nev = 12;
    cfg.dense_threshold = 0;
    cfg.eig_tol = 1e-12;
    part::DecompositionInfo info;
    info.keep_global = true;
    const core::LocalSystem sys = part::DomainDecomposer::decompose(cfg, comm, info);
    if (core::mpiRank(comm) == 0) std::cout << "case " << info.description << ", np=" << core::mpiSize(comm) << "\n";

    for (const std::string pou : {"boolean", "multiplicity"}) {
        cfg.pou = pou;
        part::Subdomain sd(sys, pou);
        linalg::DistributedMatrix A(sd);
        checkEigensolver(sd, cfg.tau, cfg.nev);
        for (const std::string ol : {"ras", "asm"})
            for (const std::string co : {"deflated", "additive"}) {
                cfg.one_level = ol;
                cfg.coarse = co;
                checkPreconditioner(sd, A, cfg, info.A_perm, pou + "/" + ol + "/" + co);
            }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    try {
        runCase("convdiff2d", 24, 1, 0.01);
        runCase("laplace2d", 20, 1, 1.0);
        runCase("coupled2d", 12, 2, 0.1);
        runCase("convdiff3d", 8, 1, 0.1);
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int f = test::failures();
    if (core::mpiRank(MPI_COMM_WORLD) == 0) std::cout << (f ? "FAILED" : "ALL PASSED") << std::endl;
    MPI_Finalize();
    return f ? 1 : 0;
}
