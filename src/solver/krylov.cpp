/**
 * @file krylov.cpp
 * @brief Stopping norms and factory of the Krylov solvers.
 */
#include "krylov.hpp"

#include "gmres.hpp"
#include "bicgstab.hpp"

#include <stdexcept>

namespace schwarz2lvl::solv {

StopNorm parseStopNorm(const std::string& s)
{
    if (s == "l2") return StopNorm::L2;
    if (s == "openfoam") return StopNorm::OpenFOAM;
    throw std::invalid_argument("unknown stopping norm '" + s + "' (l2|openfoam)");
}

const char* toString(StopNorm s) { return s == StopNorm::L2 ? "l2" : "openfoam"; }

std::unique_ptr<KrylovSolver> makeKrylovSolver(const core::SolverConfig& cfg, double rtol, double atol)
{
    const StopNorm stop = parseStopNorm(cfg.stop_norm);
    if (atol > 0.0 && stop != StopNorm::OpenFOAM)
        throw std::invalid_argument("an absolute tolerance needs the OpenFOAM stopping norm");
    if (cfg.krylov == "gmres") return std::make_unique<Gmres>(cfg.restart, rtol, cfg.maxit, stop, atol);
    if (cfg.krylov == "bicgstab") return std::make_unique<BiCgStab>(rtol, cfg.maxit, stop, atol);
    throw std::invalid_argument("unknown Krylov solver '" + cfg.krylov + "' (gmres|bicgstab)");
}

std::unique_ptr<KrylovSolver> makeKrylovSolver(const core::SolverConfig& cfg)
{
    return makeKrylovSolver(cfg, cfg.rtol, cfg.atol);
}

namespace detail {

double norm1(const core::Vec& v, MPI_Comm comm)
{
    double s = v.cwiseAbs().sum();
    MPI_Allreduce(MPI_IN_PLACE, &s, 1, MPI_DOUBLE, MPI_SUM, comm);
    return s;
}

double openfoamNormFactor(const linalg::DistributedMatrix& A, const core::Vec& b, const core::Vec& x0, const core::Vec& Ax0,
                          MPI_Comm comm)
{
    double sc[2] = {x0.sum(), static_cast<double>(x0.size())};
    MPI_Allreduce(MPI_IN_PLACE, sc, 2, MPI_DOUBLE, MPI_SUM, comm);
    const double xm = (sc[1] > 0.0) ? sc[0] / sc[1] : 0.0;
    if (xm == 0.0) return norm1(Ax0, comm) + norm1(b, comm) + 1e-20;
    core::Vec rowsum;
    A.apply(core::Vec::Ones(x0.size()), rowsum);
    return norm1(Ax0 - xm * rowsum, comm) + norm1(b - xm * rowsum, comm) + 1e-20;
}

} // namespace detail

} // namespace schwarz2lvl::solv
