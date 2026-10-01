/**
 * @file test_subdomain.cpp
 * @brief Tests of the decomposition phase: overlap, halo, A_ii, lumping sums,
 *        partition of unity and distributed matrix-vector product.
 *
 * Run with several process counts, e.g. `mpirun -np 4 ./bin/test_subdomain`.
 */
#include "config.hpp"
#include "distributed_matrix.hpp"
#include "domain_decomposer.hpp"
#include "subdomain.hpp"
#include "test_utils.hpp"

#include <set>

using namespace schwarz2lvl;

/// @brief Runs all subdomain checks for one generated problem.
void runCase(const std::string& gen, int nx, int ncomp, const std::string& pou)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = core::mpiRank(comm);
    core::SolverConfig cfg;
    cfg.generate = gen;
    cfg.nx = cfg.ny = cfg.nz = nx;
    cfg.nu = 0.01;
    cfg.ncomp = ncomp;
    cfg.pou = pou;
    part::DecompositionInfo info;
    info.keep_global = true;
    core::LocalSystem sys = part::DomainDecomposer::decompose(cfg, comm, info);
    if (rank == 0) std::cout << "case " << info.description << ", pou=" << pou << ", np=" << core::mpiSize(comm) << "\n";
    part::CsrMatrix Ag = info.A_perm;
    test::bcastCsr(Ag, comm);
    const core::SpMat A = test::toEigen(Ag);
    const core::SpMat At = A.transpose();

    part::Subdomain sd(sys, pou);
    const core::GlobalIndex b = sys.rowBegin(), e = sys.rowEnd();

    // (a) Overlap = distance-one neighbours of Omega_I in G(A + A^T).
    std::set<core::GlobalIndex> ref;
    for (core::GlobalIndex i = b; i < e; ++i) {
        for (core::SpMat::InnerIterator it(At, static_cast<core::LocalIndex>(i)); it; ++it) // row i of A
            if (it.row() < b || it.row() >= e) ref.insert(it.row());
        for (core::SpMat::InnerIterator it(A, static_cast<core::LocalIndex>(i)); it; ++it)  // row i of A^T
            if (it.row() < b || it.row() >= e) ref.insert(it.row());
    }
    const std::vector<core::GlobalIndex> refv(ref.begin(), ref.end());
    test::check(refv == sd.ghosts(), "overlap Omega_Gamma = distance-1 neighbours in G(A+A^T)");

    // (b) A_ii = A(Omega_i, Omega_i) and (c) lumping sums.
    const core::LocalIndex nl = sd.nLocal();
    core::Mat Aref = core::Mat::Zero(nl, nl);
    core::Vec sref = core::Vec::Zero(sd.nGhost());
    for (core::LocalIndex r = 0; r < nl; ++r) {
        const core::GlobalIndex gr = sd.globalId(r);
        for (std::int64_t k = Ag.row_ptr[static_cast<std::size_t>(gr)]; k < Ag.row_ptr[static_cast<std::size_t>(gr) + 1]; ++k) {
            const core::LocalIndex c = sd.localIndex(Ag.col[static_cast<std::size_t>(k)]);
            if (c >= 0)
                Aref(r, c) = Ag.val[static_cast<std::size_t>(k)];
            else if (r >= sd.nOwned())
                sref[r - sd.nOwned()] += std::abs(Ag.val[static_cast<std::size_t>(k)]);
        }
    }
    test::check((core::Mat(sd.Aii()) - Aref).norm() <= 1e-14 * Aref.norm(), "A_ii = R_i A R_i^T");
    test::check((sd.lumpingSums() - sref).norm() <= 1e-14 * (1.0 + sref.norm()), "lumping sums s_i (Def. 3.1)");

    // (d) Partition of unity: sum_i R_i^T D_i R_i = I  <=>  reverseAdd(D_i) = 1 on owned unknowns.
    core::Vec d = sd.pou();
    sd.halo().reverseAdd(d);
    test::check((d.head(sd.nOwned()).array() - 1.0).abs().maxCoeff() < 1e-14, "sum_i R_i^T D_i R_i = I");

    // (e) Distributed SpMV against the global product.
    core::Vec xg = core::Vec::LinSpaced(static_cast<Eigen::Index>(Ag.nrows), -1.0, 2.0).array().sin();
    core::Vec yg = A * xg;
    linalg::DistributedMatrix op(sd);
    core::Vec y(sd.nOwned());
    op.apply(xg.segment(b, sd.nOwned()), y);
    test::check(test::relDiff(y, yg.segment(b, sd.nOwned())) < 1e-14, "distributed SpMV");

    // (f) forward() gives R_i x.
    core::Vec xl = core::Vec::Zero(nl);
    xl.head(sd.nOwned()) = xg.segment(b, sd.nOwned());
    sd.halo().forward(xl);
    bool ok = true;
    for (core::LocalIndex l = 0; l < nl; ++l) ok = ok && (xl[l] == xg[sd.globalId(l)]);
    test::check(ok, "halo forward = restriction R_i");

    // (g) gather in original order returns the original vector.
    std::vector<core::Scalar> orig = part::DomainDecomposer::gatherOriginalOrder(sys, xg.segment(b, sd.nOwned()), info);
    ok = true;
    if (rank == 0)
        for (std::size_t k = 0; k < orig.size(); ++k) ok = ok && (orig[static_cast<std::size_t>(info.new_to_old[k])] == xg[static_cast<Eigen::Index>(k)]);
    test::check(ok, "gather in original ordering");
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    try {
        runCase("convdiff2d", 16, 1, "boolean");
        runCase("convdiff2d", 16, 1, "multiplicity");
        runCase("coupled2d", 10, 3, "multiplicity");
        runCase("laplace3d", 7, 1, "boolean");
    } catch (const std::exception& e) {
        std::cerr << "exception: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int f = test::failures();
    if (core::mpiRank(MPI_COMM_WORLD) == 0) std::cout << (f ? "FAILED" : "ALL PASSED") << std::endl;
    MPI_Finalize();
    return f ? 1 : 0;
}
