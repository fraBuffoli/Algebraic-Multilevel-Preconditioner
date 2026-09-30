/**
 * @file coarse_solver.cpp
 * @brief Coarse solvers: distributed MUMPS and a gather-to-root fallback.
 */
#include "coarse_solver.hpp"

#include "local_solver.hpp"
#include "logger.hpp"
#include "mpi_utils.hpp"
#include "timer.hpp"

#ifdef USE_MUMPS
#include <dmumps_c.h>
#endif

#include <algorithm>
#include <climits>
#include <stdexcept>

namespace schwarz2lvl::prec {


/// @brief Counts/displacements of the coarse blocks (for Gatherv/Scatterv).
void coarseCounts(const std::vector<core::GlobalIndex>& off, std::vector<int>& cnt, std::vector<int>& dsp)
{
    const std::size_t p = off.size() - 1;
    cnt.resize(p);
    for (std::size_t k = 0; k < p; ++k) cnt[k] = static_cast<int>(off[k + 1] - off[k]);
    dsp = core::displacements(cnt);
}

/// @brief Splits triplets into separate arrays and gathers them on rank @p root of @p comm.
void gatherTriplets(const std::vector<core::GlobalTriplet>& t, MPI_Comm comm, int root, std::vector<core::GlobalIndex>& rows,
                    std::vector<core::GlobalIndex>& cols, std::vector<core::Scalar>& vals)
{
    const int np = core::mpiSize(comm);
    std::vector<core::GlobalIndex> r(t.size()), c(t.size());
    std::vector<core::Scalar> v(t.size());
    for (std::size_t k = 0; k < t.size(); ++k) {
        r[k] = t[k].row;
        c[k] = t[k].col;
        v[k] = t[k].val;
    }
    int mine = static_cast<int>(t.size());
    std::vector<int> cnt(static_cast<std::size_t>(np));
    MPI_Gather(&mine, 1, MPI_INT, cnt.data(), 1, MPI_INT, root, comm);
    std::vector<int> dsp = core::displacements(cnt);
    const bool is_root = core::mpiRank(comm) == root;
    const std::size_t tot = is_root ? static_cast<std::size_t>(dsp.back() + cnt.back()) : 0;
    rows.resize(tot);
    cols.resize(tot);
    vals.resize(tot);
    MPI_Gatherv(r.data(), mine, core::mpiType<core::GlobalIndex>(), rows.data(), cnt.data(), dsp.data(), core::mpiType<core::GlobalIndex>(),
                root, comm);
    MPI_Gatherv(c.data(), mine, core::mpiType<core::GlobalIndex>(), cols.data(), cnt.data(), dsp.data(), core::mpiType<core::GlobalIndex>(),
                root, comm);
    MPI_Gatherv(v.data(), mine, MPI_DOUBLE, vals.data(), cnt.data(), dsp.data(), MPI_DOUBLE, root, comm);
}

// ------------------------------------------------------------------ root solver
/**
 * @brief Gathers A_00 on rank 0 and factorizes it with a sequential LU.
 *        Simple and robust, but not scalable: a fallback and a reference.
 */
class RootCoarseSolver final : public CoarseSolver {
public:
    explicit RootCoarseSolver(std::string backend) : backend_(std::move(backend)) {}

    void setup(MPI_Comm comm, const std::vector<core::GlobalIndex>& offsets, std::vector<core::GlobalTriplet>& entries,
               bool /*symmetric*/) override
    {
        comm_ = comm;
        offsets_ = offsets;
        coarseCounts(offsets_, cnt_, dsp_);
        std::vector<core::GlobalIndex> r, c;
        std::vector<core::Scalar> v;
        gatherTriplets(entries, comm_, 0, r, c, v);
        int ok = 1;
        std::string err;
        if (core::mpiRank(comm_) == 0) {
            const auto n0 = static_cast<core::LocalIndex>(offsets_.back());
            std::vector<core::Triplet> t;
            t.reserve(r.size());
            for (std::size_t k = 0; k < r.size(); ++k)
                t.emplace_back(static_cast<core::LocalIndex>(r[k]), static_cast<core::LocalIndex>(c[k]), v[k]);
            core::SpMat E(n0, n0);
            E.setFromTriplets(t.begin(), t.end()); // duplicates are summed
            solver_ = linalg::makeLocalSolver(backend_);
            ok = solver_->factorize(E) ? 1 : 0;
            if (!ok) err = solver_->lastError();
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, comm_);
        if (!ok) throw std::runtime_error("coarse matrix factorization failed (root solver) " + err);
    }

    void solve(const core::Vec& c, core::Vec& y) const override
    {
        const bool root = core::mpiRank(comm_) == 0;
        if (root) buf_.resize(static_cast<Eigen::Index>(offsets_.back()));
        MPI_Gatherv(c.data(), static_cast<int>(c.size()), MPI_DOUBLE, buf_.data(), cnt_.data(), dsp_.data(),
                    MPI_DOUBLE, 0, comm_);
        if (root) {
            solver_->solve(buf_, sol_);
            buf_ = sol_;
        }
        y.resize(c.size());
        MPI_Scatterv(buf_.data(), cnt_.data(), dsp_.data(), MPI_DOUBLE, y.data(), static_cast<int>(y.size()),
                     MPI_DOUBLE, 0, comm_);
    }

    std::string name() const override { return "root(" + backend_ + ")"; }

private:
    std::string backend_;
    MPI_Comm comm_ = MPI_COMM_NULL;
    std::vector<core::GlobalIndex> offsets_;
    std::vector<int> cnt_, dsp_;
    std::unique_ptr<linalg::LocalSolver> solver_;
    mutable core::Vec buf_, sol_;
};

#ifdef USE_MUMPS
// ------------------------------------------------------------------------ MUMPS
/**
 * @brief Distributed multifrontal LU (or LDL^T) of A_00 with MUMPS.
 *
 * - Distributed assembled input (ICNTL(18) = 3): every MUMPS process passes
 *   its own coefficients, MUMPS sums duplicates.
 * - Optionally only a subset of `coarse_procs` processes (the leaders of
 *   consecutive groups of ranks, e.g. one or a few per node) runs MUMPS: the
 *   other ranks send their coefficients to their group leader. This limits
 *   the communication cost of the coarse solve at large process counts.
 * - Centralized right-hand side and solution on the host (world rank 0):
 *   one MPI_Gatherv / MPI_Scatterv per application.
 * - SYM = 2 (general symmetric LDL^T) if A is symmetric, SYM = 0 otherwise.
 */
class MumpsCoarseSolver final : public CoarseSolver {
public:
    explicit MumpsCoarseSolver(int coarse_procs) : coarse_procs_(coarse_procs) {}

    ~MumpsCoarseSolver() override
    {
        if (initialized_) {
            id_.job = -2;
            dmumps_c(&id_);
        }
        if (mumps_comm_ != MPI_COMM_NULL) MPI_Comm_free(&mumps_comm_);
        if (group_comm_ != MPI_COMM_NULL) MPI_Comm_free(&group_comm_);
    }

    void setup(MPI_Comm comm, const std::vector<core::GlobalIndex>& offsets, std::vector<core::GlobalTriplet>& entries,
               bool symmetric) override
    {
        comm_ = comm;
        offsets_ = offsets;
        coarseCounts(offsets_, cnt_, dsp_);
        const int rank = core::mpiRank(comm_), np = core::mpiSize(comm_);
        if (offsets_.back() > INT_MAX) throw std::runtime_error("coarse problem too large for 32-bit MUMPS");

        // Groups of `stride` consecutive ranks; the first rank of each group is a MUMPS process.
        const int nproc = (coarse_procs_ <= 0 || coarse_procs_ > np) ? np : coarse_procs_;
        const int stride = (np + nproc - 1) / nproc;
        const int group = rank / stride;
        leader_ = (rank % stride == 0);
        MPI_Comm_split(comm_, group, rank, &group_comm_);
        MPI_Comm_split(comm_, leader_ ? 0 : MPI_UNDEFINED, rank, &mumps_comm_);

        if (symmetric) // SYM = 2: only one triangle must be provided
            entries.erase(std::remove_if(entries.begin(), entries.end(),
                                         [](const core::GlobalTriplet& t) { return t.row < t.col; }),
                          entries.end());
        std::vector<core::GlobalIndex> r, c;
        gatherTriplets(entries, group_comm_, 0, r, c, a_);
        irn_.resize(r.size());
        jcn_.resize(c.size());
        for (std::size_t k = 0; k < r.size(); ++k) { // MUMPS uses 1-based indices
            irn_[k] = static_cast<MUMPS_INT>(r[k] + 1);
            jcn_[k] = static_cast<MUMPS_INT>(c[k] + 1);
        }

        int status = 0;
        if (leader_) {
            id_.comm_fortran = static_cast<MUMPS_INT>(MPI_Comm_c2f(mumps_comm_));
            id_.par = 1;                   // the host takes part in the factorization
            id_.sym = symmetric ? 2 : 0;
            id_.job = -1;
            dmumps_c(&id_);
            initialized_ = true;
            id_.icntl[0] = 6;              // error messages on stdout
            id_.icntl[1] = -1;             // no diagnostics
            id_.icntl[2] = -1;             // no global information
            id_.icntl[3] = 1;              // print errors only
            id_.icntl[17] = 3;             // ICNTL(18): distributed assembled matrix
            id_.icntl[19] = 0;             // ICNTL(20): dense centralized RHS
            id_.icntl[20] = 0;             // ICNTL(21): centralized solution
            id_.n = static_cast<MUMPS_INT>(offsets_.back());
            id_.nnz_loc = static_cast<MUMPS_INT8>(irn_.size());
            id_.irn_loc = irn_.data();
            id_.jcn_loc = jcn_.data();
            id_.a_loc = a_.data();
            for (int attempt = 0; attempt < 5; ++attempt) {
                id_.job = 4;               // analysis + factorization
                dmumps_c(&id_);
                status = id_.infog[0];
                // -9 / -8 / -14 / -15 ... : workspace too small -> more relaxation (ICNTL(14)).
                if (status == -9 || status == -8 || status == -14 || status == -15 || status == -17 ||
                    status == -20) {
                    id_.icntl[13] = std::max(40, 2 * id_.icntl[13]);
                    continue;
                }
                break;
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, comm_);
        if (status < 0)
            throw std::runtime_error("MUMPS factorization of the coarse matrix failed, INFOG(1) = " +
                                     std::to_string(status));
    }

    void solve(const core::Vec& c, core::Vec& y) const override
    {
        const int rank = core::mpiRank(comm_);
        if (rank == 0) rhs_.resize(static_cast<std::size_t>(offsets_.back()));
        MPI_Gatherv(c.data(), static_cast<int>(c.size()), MPI_DOUBLE, rhs_.data(), cnt_.data(), dsp_.data(),
                    MPI_DOUBLE, 0, comm_);
        if (leader_) {
            id_.job = 3;
            id_.nrhs = 1;
            id_.lrhs = id_.n;
            id_.rhs = (rank == 0) ? rhs_.data() : nullptr;
            dmumps_c(&id_);
        }
        y.resize(c.size());
        MPI_Scatterv(rhs_.data(), cnt_.data(), dsp_.data(), MPI_DOUBLE, y.data(), static_cast<int>(y.size()),
                     MPI_DOUBLE, 0, comm_);
    }

    std::string name() const override { return "mumps"; }

private:
    int coarse_procs_;
    MPI_Comm comm_ = MPI_COMM_NULL, group_comm_ = MPI_COMM_NULL, mumps_comm_ = MPI_COMM_NULL;
    bool leader_ = false, initialized_ = false;
    std::vector<core::GlobalIndex> offsets_;
    std::vector<int> cnt_, dsp_;
    std::vector<MUMPS_INT> irn_, jcn_;
    std::vector<core::Scalar> a_;
    mutable DMUMPS_STRUC_C id_{};
    mutable std::vector<core::Scalar> rhs_;
};
#endif

std::unique_ptr<CoarseSolver> makeCoarseSolver(const std::string& name, int coarse_procs,
                                               const std::string& local_solver)
{
#ifdef USE_MUMPS
    if (name == "auto" || name == "mumps") return std::make_unique<MumpsCoarseSolver>(coarse_procs);
#else
    if (name == "mumps") throw std::invalid_argument("coarse solver 'mumps' not compiled in (USE_MUMPS=1)");
    if (name == "auto") return std::make_unique<RootCoarseSolver>(local_solver);
#endif
    if (name == "root") return std::make_unique<RootCoarseSolver>(local_solver);
    throw std::invalid_argument("unknown coarse solver '" + name + "'");
}

} // namespace schwarz2lvl
