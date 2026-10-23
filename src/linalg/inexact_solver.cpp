/**
 * @file inexact_solver.cpp
 * @brief Approximate inverse of A_ii (P7): one AMG V-cycle (AMGCL).
 */
#include "inexact_solver.hpp"

#ifdef USE_AMGCL
#ifndef AMGCL_NO_BOOST
#define AMGCL_NO_BOOST // runtime (property tree) interface not used: no Boost needed
#endif
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/util.hpp>
#endif

#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace schwarz2lvl::linalg {

#ifdef USE_AMGCL
/**
 * @brief One AMG V-cycle as approximate inverse: AMGCL, builtin backend (sequential: one MPI rank
 *        per core, no OpenMP), value type double or float, Gauss-Seidel smoother.
 *
 * Setup: A_ii is converted to CSR (Eigen row-major; for float the values are rounded) and copied
 * by AMGCL. Coarsening: smoothed aggregation or Ruge-Stuben. Smoother on every level: Gauss-Seidel,
 * forward before and backward after the coarse correction (the cycle stays symmetric),
 * AmgOptions::sweeps steps each.
 *
 * Coarsest level: AMGCL stops coarsening when a level has at most `coarse_enough` unknowns and
 * factorizes it with a skyline (profile) LU. The AMGCL default, 3000, was a trap: a level of
 * 1000-3000 unknowns of a 3D coarse operator has a nearly dense profile, so its factorization
 * dominated the setup and its triangular solves every V-cycle (G100, 17k cells per subdomain:
 * 2 levels, AMG setup 0.55 s, more than the exact Cholesky factorization). With 500 the hierarchy
 * gets one more level (locally, 16k cells: setup 0.21 -> 0.03 s, V-cycle 4.6 -> 1.6 ms, +1 GMRES
 * iteration).
 */
template <typename Real, template <class> class Coarsening> class AmgSolver final : public LocalSolver {
    using Backend = amgcl::backend::builtin<Real>;
    using AMG = amgcl::amg<Backend, Coarsening, amgcl::relaxation::gauss_seidel>;
    using RowMat = Eigen::SparseMatrix<Real, Eigen::RowMajor, core::LocalIndex>;

public:
    AmgSolver(const AmgOptions& o, std::string label) : opt_(o), label_(std::move(label))
    {
        prm_.npre = prm_.npost = static_cast<unsigned>(o.sweeps);
        prm_.pre_cycles = 1; // one V-cycle per application
        prm_.coarse_enough = static_cast<unsigned>(o.coarse_enough);
        prm_.allow_rebuild = o.reuse; // keeps P and R for rebuild() only when they are reused
    }

    bool factorize(const core::SpMat& A) override
    {
        try {
            const RowMat R = A.template cast<Real>();
            amg_ = std::make_unique<AMG>(crs(R), prm_);
        } catch (const std::exception& e) {
            amg_.reset();
            error_ = label_ + ": setup failed: " + e.what();
            return false;
        }
        n_ = static_cast<std::size_t>(A.rows());
        f_.assign(n_, Real(0));
        u_.assign(n_, Real(0));
        describe();
        return true;
    }
    /// Full setup, or with AmgOptions::reuse AMGCL rebuild() (transfer operators of the last setup
    /// kept, Galerkin products and smoothers recomputed).
    bool refactorize(const core::SpMat& A) override
    {
        fell_back_ = false;
        if (!amg_ || !opt_.reuse) return factorize(A);
        try {
            const RowMat R = A.template cast<Real>();
            amg_->rebuild(crs(R));
            return true;
        } catch (const std::exception&) {
            fell_back_ = true;
            return factorize(A);
        }
    }
    void solve(const core::Vec& b, core::Vec& x) const override
    {
        for (std::size_t i = 0; i < n_; ++i) f_[i] = static_cast<Real>(b[static_cast<Eigen::Index>(i)]);
        amg_->apply(f_, u_); // u = 0, then one V-cycle: a fixed linear operator
        x.resize(static_cast<Eigen::Index>(n_));
        for (std::size_t i = 0; i < n_; ++i) x[static_cast<Eigen::Index>(i)] = static_cast<double>(u_[i]);
    }
    std::string name() const override
    {
        return label_ + " V(" + std::to_string(opt_.sweeps) + "," + std::to_string(opt_.sweeps) + "), coarsest <= " +
               std::to_string(opt_.coarse_enough) + (opt_.reuse ? ", reused hierarchy" : "") +
               (std::is_same_v<Real, float> ? " single" : "") + info_;
    }

private:
    AmgOptions opt_;
    std::string label_;            ///< "amg <coarsening>+gs".
    typename AMG::params prm_;
    std::unique_ptr<AMG> amg_;
    std::size_t n_ = 0;
    mutable std::vector<Real> f_;  ///< Right-hand side in precision Real.
    mutable std::vector<Real> u_;  ///< Solution in precision Real.
    std::string info_;             ///< " [levels, operator complexity]" of the last setup.

    static auto crs(const RowMat& R)
    {
        const auto n = static_cast<std::size_t>(R.rows());
        const auto nnz = static_cast<std::size_t>(R.nonZeros());
        return std::make_tuple(n, amgcl::make_iterator_range(R.outerIndexPtr(), R.outerIndexPtr() + n + 1),
                               amgcl::make_iterator_range(R.innerIndexPtr(), R.innerIndexPtr() + nnz),
                               amgcl::make_iterator_range(R.valuePtr(), R.valuePtr() + nnz));
    }
    /// Levels and operator complexity (sum of the nonzeros of all levels / nnz(A_ii)), from
    /// the summary that AMGCL prints.
    void describe()
    {
        std::ostringstream os;
        os << *amg_;
        std::istringstream is(os.str());
        std::string line, levels, oc;
        while (std::getline(is, line)) {
            const auto c = line.find(':');
            if (c == std::string::npos) continue;
            std::istringstream v(line.substr(c + 1));
            if (line.rfind("Number of levels", 0) == 0) v >> levels;
            if (line.rfind("Operator complexity", 0) == 0) v >> oc;
        }
        info_ = " [" + levels + " levels, op. complexity " + oc + "]";
    }
};

namespace {

template <typename Real> std::unique_ptr<LocalSolver> makeAmg(const AmgOptions& o)
{
    const std::string label = "amg " + o.coarsening + "+gs";
    if (o.coarsening == "sa")
        return std::make_unique<AmgSolver<Real, amgcl::coarsening::smoothed_aggregation>>(o, label);
    if (o.coarsening == "rs") return std::make_unique<AmgSolver<Real, amgcl::coarsening::ruge_stuben>>(o, label);
    throw std::invalid_argument("unknown AMG coarsening '" + o.coarsening + "' (sa|rs)");
}

} // namespace
#endif // USE_AMGCL

bool amgAvailable()
{
#ifdef USE_AMGCL
    return true;
#else
    return false;
#endif
}

std::unique_ptr<LocalSolver> makeInexactSolver(const LocalSolverSpec& spec)
{
    if (spec.approx != "amg")
        throw std::invalid_argument("unknown local approximation '" + spec.approx + "' (none|amg)");
    if (spec.amg.sweeps < 1) throw std::invalid_argument("AMG sweeps must be >= 1");
    if (spec.amg.coarse_enough < 1) throw std::invalid_argument("AMG coarse_enough must be >= 1");
#ifdef USE_AMGCL
    if (spec.single) return makeAmg<float>(spec.amg);
    return makeAmg<double>(spec.amg);
#else
    throw std::invalid_argument("--local-approx amg: this build has no AMGCL (make USE_AMGCL=1)");
#endif
}

} // namespace schwarz2lvl::linalg
