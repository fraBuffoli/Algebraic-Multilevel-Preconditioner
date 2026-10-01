/**
 * @file main.cpp
 * @brief Driver: decomposition, preconditioner setup, GMRES solve and reporting.
 *
 * Example:
 * @code
 * mpirun -np 16 ./bin/schwarz2lvl --generate convdiff3d --nx 40 --nu 1e-2 --tau 0.3 --nev 60
 * mpirun -np 64 ./bin/schwarz2lvl --matrix matrices/G3_circuit.mtx --csv results.csv
 * @endcode
 */
#include "config.hpp"
#include "distributed_matrix.hpp"
#include "domain_decomposer.hpp"
#include "gmres.hpp"
#include "local_solver.hpp"
#include "logger.hpp"
#include "matrix_market_io.hpp"
#include "mpi_utils.hpp"
#include "one_level_schwarz.hpp"
#include "subdomain.hpp"
#include "timer.hpp"
#include "two_level_schwarz.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

using namespace schwarz2lvl;

/// @brief Prints min/avg/max of an integer quantity over the ranks.
void printMinAvgMax(MPI_Comm comm, const std::string& what, long long v)
{
    long long mn = 0, mx = 0, sum = 0;
    MPI_Allreduce(&v, &mn, 1, MPI_LONG_LONG, MPI_MIN, comm);
    MPI_Allreduce(&v, &mx, 1, MPI_LONG_LONG, MPI_MAX, comm);
    MPI_Allreduce(&v, &sum, 1, MPI_LONG_LONG, MPI_SUM, comm);
    utils::Log::root(comm) << "  " << std::left << std::setw(27) << what << mn << " / " << std::fixed
                    << std::setprecision(1) << static_cast<double>(sum) / core::mpiSize(comm) << " / " << mx
                    << std::defaultfloat << "\n";
}

/// @brief The actual program (exceptions are handled by main()).
int run(int argc, char** argv)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = core::mpiRank(comm);
    const int np = core::mpiSize(comm);

    core::SolverConfig cfg;
    try {
        cfg = core::SolverConfig::fromCommandLine(argc, argv);
    } catch (const std::invalid_argument& e) {
        if (rank == 0) {
            if (std::string(e.what()) != "help") std::cerr << "error: " << e.what() << "\n\n";
            std::cout << core::SolverConfig::help();
        }
        return std::string(e.what()) == "help" ? 0 : 1;
    }
    utils::Log::setVerbosity(cfg.verbose);
    std::ostream& out = utils::Log::root(comm);
    out << "schwarz2lvl: two-level algebraic Schwarz (Al Daas, Jolivet, Rees, SISC 2023), " << np
        << " MPI processes\n";
    if (rank == 0 && cfg.verbose >= 1) cfg.print(out);

    // ---------------------------------------------------------------- decomposition
    part::DecompositionInfo info;
    core::LocalSystem sys = part::DomainDecomposer::decompose(cfg, comm, info);
    out << "Problem\n  " << info.description << "\n  block size " << sys.block_size
        << (sys.symmetric ? ", symmetric" : ", nonsymmetric") << "\n";

    // ---------------------------------------------------------------- setup
    MPI_Barrier(comm);
    const double t_setup0 = MPI_Wtime();
    part::Subdomain sd(sys, cfg.pou);
    linalg::DistributedMatrix A(sd);
    out << "Subdomains (min / avg / max)\n";
    printMinAvgMax(comm, "owned unknowns n_I", sd.nOwned());
    printMinAvgMax(comm, "overlap unknowns n_Gamma", sd.nGhost());
    printMinAvgMax(comm, "neighbours", sd.halo().numNeighbors());
    printMinAvgMax(comm, "nnz(A_ii)", sd.Aii().nonZeros());

    std::unique_ptr<prec::Preconditioner> M;
    const prec::TwoLevelSchwarz* two = nullptr;
    if (cfg.coarse == "none") {
        utils::ScopedTimer t("setup");
        M = std::make_unique<prec::OneLevelSchwarz>(
            sd, cfg.one_level == "asm" ? prec::OneLevelSchwarz::Variant::ASM : prec::OneLevelSchwarz::Variant::RAS,
            cfg.local_solver);
    } else {
        auto p = std::make_unique<prec::TwoLevelSchwarz>(sd, A, cfg);
        two = p.get();
        M = std::move(p);
    }
    MPI_Barrier(comm);
    const double t_setup = MPI_Wtime() - t_setup0;
    if (two) two->printStats(out);
    out << "Preconditioner " << M->name() << ", setup " << t_setup << " s\n";

    // ---------------------------------------------------------------- solve
    const core::Vec b = Eigen::Map<const core::Vec>(sys.rhs.data(), static_cast<Eigen::Index>(sys.rhs.size()));
    core::Vec x = core::Vec::Zero(b.size()); // zero initial guess (Section 4)
    MPI_Barrier(comm);
    const double t_solve0 = MPI_Wtime();
    const solv::GmresResult res = solv::Gmres(cfg.restart, cfg.rtol, cfg.maxit).solve(A, *M, b, x);
    MPI_Barrier(comm);
    const double t_solve = MPI_Wtime() - t_solve0;

    out << "GMRES(" << cfg.restart << ")\n"
        << "  " << (res.converged ? "converged" : "NOT converged") << " in " << res.iterations << " iterations\n"
        << std::scientific << std::setprecision(3) << "  relative residual (GMRES)  " << res.rel_residual << "\n"
        << "  relative residual (true)   " << res.true_rel_residual << "\n"
        << std::defaultfloat << "  solve time " << t_solve << " s, total (setup+solve) " << t_setup + t_solve
        << " s\n";
    // Collective: every rank must call it, only rank 0 prints.
    utils::TimerRegistry::instance().report(comm, cfg.verbose >= 1 ? out : utils::Log::nullStream());

    // ---------------------------------------------------------------- output
    const double t_eig = utils::TimerRegistry::instance().maxOverRanks(comm, "setup.eigensolver");
    const double t_asm = utils::TimerRegistry::instance().maxOverRanks(comm, "setup.coarse_assembly");
    const double t_cf = utils::TimerRegistry::instance().maxOverRanks(comm, "setup.coarse_factorization");
    if (!cfg.write_solution.empty()) {
        const std::vector<core::Scalar> xg = part::DomainDecomposer::gatherOriginalOrder(sys, x, info);
        if (rank == 0) part::MatrixMarketIO::writeVector(cfg.write_solution, xg);
    }
    if (rank == 0) {
        if (!cfg.history.empty()) {
            std::ofstream h(cfg.history);
            h << std::scientific << std::setprecision(10);
            for (std::size_t k = 0; k < res.history.size(); ++k) h << k << " " << res.history[k] << "\n";
        }
        if (!cfg.csv.empty()) {
            std::ifstream probe(cfg.csv);
            const bool header = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
            probe.close();
            std::ofstream c(cfg.csv, std::ios::app);
            if (header)
                c << "problem,n,nnz,np,pou,one_level,coarse,tau,nev,local_solver,coarse_solver,n0,m_min,m_max,"
                     "iterations,converged,true_rel_res,t_setup,t_eig,t_coarse_asm,t_coarse_fact,t_solve\n";
            c << '"' << info.description << '"' << ',' << sys.n_global << ',' << info.nnz << ',' << np << ','
              << cfg.pou << ',' << cfg.one_level << ',' << cfg.coarse << ',' << cfg.tau << ',' << cfg.nev << ','
              << cfg.local_solver << ',' << cfg.coarse_solver << ',' << (two ? two->stats().n0 : 0) << ','
              << (two ? two->stats().m_min : 0) << ',' << (two ? two->stats().m_max : 0) << ',' << res.iterations
              << ',' << (res.converged ? 1 : 0) << ',' << res.true_rel_residual << ',' << t_setup << ',' << t_eig
              << ',' << t_asm << ',' << t_cf << ',' << t_solve << "\n";
        }
    }
    return res.converged ? 0 : 2;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int code = 0;
    try {
        code = run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "[rank " << core::mpiRank(MPI_COMM_WORLD) << "] error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return code;
}
