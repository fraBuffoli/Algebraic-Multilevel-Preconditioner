/**
 * @file coarse_solver.cpp
 * @brief Coarse solvers: distributed MUMPS, distributed MKL PARDISO (cluster_sparse_solver)
 *        and a gather-to-root fallback.
 */
#include "coarse_solver.hpp"

#include "local_solver.hpp"
#include "logger.hpp"
#include "mpi_utils.hpp"
#include "timer.hpp"

#ifdef USE_MUMPS
#include <dmumps_c.h>
#endif
#ifdef USE_CPARDISO
#include <mkl_cluster_sparse_solver.h>
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

/// @brief Gathers only the values of the triplets on rank @p root, in the same order as gatherTriplets().
std::vector<core::Scalar> gatherValues(const std::vector<core::GlobalTriplet>& t, MPI_Comm comm, int root)
{
    const int np = core::mpiSize(comm);
    std::vector<core::Scalar> v(t.size());
    for (std::size_t k = 0; k < t.size(); ++k) v[k] = t[k].val;
    int mine = static_cast<int>(t.size());
    std::vector<int> cnt(static_cast<std::size_t>(np));
    MPI_Gather(&mine, 1, MPI_INT, cnt.data(), 1, MPI_INT, root, comm);
    const std::vector<int> dsp = core::displacements(cnt);
    const bool is_root = core::mpiRank(comm) == root;
    std::vector<core::Scalar> vals(is_root ? static_cast<std::size_t>(dsp.back() + cnt.back()) : 0);
    MPI_Gatherv(v.data(), mine, MPI_DOUBLE, vals.data(), cnt.data(), dsp.data(), MPI_DOUBLE, root, comm);
    return vals;
}

// ------------------------------------------------------------------ root solver
/**
 * @brief Gathers A_00 on rank 0 and factorizes it with a sequential LU.
 *        Simple and robust, but not scalable: a fallback and a reference.
 *
 * update(): rank 0 keeps the assembled matrix and, for every gathered coefficient, its
 * position in the value array (duplicates share a position), so new values are summed in
 * place and the LU is redone with LocalSolver::refactorize().
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
            E_.resize(n0, n0);
            E_.setFromTriplets(t.begin(), t.end()); // duplicates are summed
            // Position of every gathered coefficient in E_.valuePtr() (column-major, compressed).
            pos_.resize(r.size());
            const core::LocalIndex* inner = E_.innerIndexPtr();
            for (std::size_t k = 0; k < r.size(); ++k) {
                const core::LocalIndex* b = inner + E_.outerIndexPtr()[c[k]];
                const core::LocalIndex* e = inner + E_.outerIndexPtr()[c[k] + 1];
                pos_[k] = static_cast<core::LocalIndex>(std::lower_bound(b, e, static_cast<core::LocalIndex>(r[k])) - inner);
            }
            solver_ = linalg::makeLocalSolver(backend_);
            ok = solver_->factorize(E_) ? 1 : 0;
            if (!ok) err = solver_->lastError();
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, comm_);
        if (!ok) throw std::runtime_error("coarse matrix factorization failed (root solver) " + err);
    }

    void update(std::vector<core::GlobalTriplet>& entries) override
    {
        const std::vector<core::Scalar> v = gatherValues(entries, comm_, 0);
        int ok = 1;
        std::string err;
        if (core::mpiRank(comm_) == 0) {
            if (v.size() != pos_.size()) throw std::logic_error("root coarse solver: the coarse pattern changed");
            core::Scalar* e = E_.valuePtr();
            std::fill(e, e + E_.nonZeros(), 0.0);
            for (std::size_t k = 0; k < v.size(); ++k) e[pos_[k]] += v[k];
            ok = solver_->refactorize(E_) ? 1 : 0;
            if (!ok) err = solver_->lastError();
            if (solver_->refactorizationFellBack()) ++fallbacks_;
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, comm_);
        if (!ok) throw std::runtime_error("coarse matrix refactorization failed (root solver) " + err);
    }

    long fallbacks() const override { return fallbacks_; } // rank 0 only (max over ranks when reported)

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
    core::SpMat E_;                     ///< Rank 0: assembled A_00 (kept for update()).
    std::vector<core::LocalIndex> pos_; ///< Rank 0: gathered coefficient k -> position in E_.valuePtr().
    long fallbacks_ = 0;                ///< Rank 0: refactorizations that fell back to a full LU.
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
 * - update(): the values are gathered on the leaders in the order of setup() (same irn/jcn)
 *   and MUMPS is called with JOB = 2 (numerical factorization, analysis of JOB = 4 kept).
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
        symmetric_ = symmetric;
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

        filter(entries);
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
            status = run(4);               // analysis + factorization
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, comm_);
        if (status < 0)
            throw std::runtime_error("MUMPS factorization of the coarse matrix failed, INFOG(1) = " +
                                     std::to_string(status));
    }

    void update(std::vector<core::GlobalTriplet>& entries) override
    {
        filter(entries);
        std::vector<core::Scalar> v = gatherValues(entries, group_comm_, 0);
        int status = 0;
        if (leader_) {
            if (v.size() != a_.size()) throw std::logic_error("MUMPS coarse solver: the coarse pattern changed");
            a_ = std::move(v);
            id_.a_loc = a_.data();
            status = run(2);               // numerical factorization only (analysis of setup() kept)
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, comm_);
        if (status < 0)
            throw std::runtime_error("MUMPS refactorization of the coarse matrix failed, INFOG(1) = " +
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
    bool symmetric_ = false;
    MPI_Comm comm_ = MPI_COMM_NULL, group_comm_ = MPI_COMM_NULL, mumps_comm_ = MPI_COMM_NULL;
    bool leader_ = false, initialized_ = false;
    std::vector<core::GlobalIndex> offsets_;
    std::vector<int> cnt_, dsp_;
    std::vector<MUMPS_INT> irn_, jcn_;
    std::vector<core::Scalar> a_;
    mutable DMUMPS_STRUC_C id_{};
    mutable std::vector<core::Scalar> rhs_;

    /// @brief SYM = 2: only one triangle must be provided (same filter in setup() and update()).
    void filter(std::vector<core::GlobalTriplet>& entries) const
    {
        if (symmetric_)
            entries.erase(std::remove_if(entries.begin(), entries.end(),
                                         [](const core::GlobalTriplet& t) { return t.row < t.col; }),
                          entries.end());
    }

    /// @brief Calls MUMPS with JOB = @p job; retries with more workspace on the "too small" errors.
    int run(int job)
    {
        int status = 0;
        for (int attempt = 0; attempt < 5; ++attempt) {
            id_.job = job;
            dmumps_c(&id_);
            status = id_.infog[0];
            // -9 / -8 / -14 / -15 ... : workspace too small -> more relaxation (ICNTL(14)).
            if (status == -9 || status == -8 || status == -14 || status == -15 || status == -17 || status == -20) {
                id_.icntl[13] = std::max(40, 2 * id_.icntl[13]);
                continue;
            }
            break;
        }
        return status;
    }
};
#endif

#ifdef USE_CPARDISO
// ----------------------------------------------------- MKL PARDISO for clusters
/**
 * @brief Distributed LU (or symmetric indefinite LDL^T) of A_00 with the MPI version of
 *        MKL PARDISO (cluster_sparse_solver), available wherever MKL is.
 *
 * - The ranks are split into `coarse_procs` groups of consecutive ranks (as for MUMPS);
 *   the leader of each group (its first rank) owns the coarse rows of the whole group,
 *   [offsets[first rank], offsets[last rank + 1]), which is a contiguous range.
 * - Every coefficient is sent to the leader owning its row, duplicates are summed, and
 *   each leader passes its rows in distributed CSR format (iparm[39] = 2): matrix,
 *   right-hand side and solution are distributed by rows, so the solve only needs a
 *   Gatherv/Scatterv inside each group (no global gather on rank 0).
 * - Only the leaders with at least one coarse row take part in the factorization.
 * - mtype = -2 (symmetric indefinite, upper triangle) if A is symmetric, 11 otherwise,
 *   with the same pivoting/scaling/refinement options as the sequential PARDISO backend
 *   (no iterative refinement unless pivots were perturbed). With Open MPI
 *   the symmetric types crash inside MKL (MKL 2023.2, Open MPI 4.1) on any communicator
 *   other than MPI_COMM_WORLD, so there symmetric matrices are factorized as general ones.
 * - Fill-in reducing ordering: METIS on the first process (iparm[1] = 2). The MPI nested
 *   dissection of PARDISO (iparm[1] = 10) hung in the tests (MKL 2023.2), so it is not used.
 * - update(): setup() keeps the communication pattern (send position of every coefficient,
 *   counts/displacements of the Alltoallv) and, on the leaders, the CSR position of every
 *   received coefficient. New values travel with one Alltoallv of doubles, are summed in
 *   place, and only phase 22 (numerical factorization) is run. If the factorization fails or
 *   perturbs more pivots than the full one (scaling/matching computed in phase 11 from the
 *   old values), phases 11 + 22 are redone.
 */
class CPardisoCoarseSolver final : public CoarseSolver {
public:
    explicit CPardisoCoarseSolver(int coarse_procs) : coarse_procs_(coarse_procs) {}

    ~CPardisoCoarseSolver() override
    {
        if (factorized_) {
            MKL_INT phase = -1; // release all internal memory
            call(phase, nullptr, nullptr);
        }
        if (solver_comm_ != MPI_COMM_NULL) MPI_Comm_free(&solver_comm_);
        if (group_comm_ != MPI_COMM_NULL) MPI_Comm_free(&group_comm_);
    }

    void setup(MPI_Comm comm, const std::vector<core::GlobalIndex>& offsets, std::vector<core::GlobalTriplet>& entries,
               bool symmetric) override
    {
        comm_ = comm;
        const int rank = core::mpiRank(comm_), np = core::mpiSize(comm_);
        if (offsets.back() > INT_MAX) throw std::runtime_error("coarse problem too large for 32-bit MKL PARDISO");
        n0_ = static_cast<MKL_INT>(offsets.back());

        // ---- groups of `stride` consecutive ranks; leaders own the rows of their group.
        const int nproc = (coarse_procs_ <= 0 || coarse_procs_ > np) ? np : coarse_procs_;
        stride_ = (np + nproc - 1) / nproc;
        const int group = rank / stride_;
        const int g0 = group * stride_, g1 = std::min(g0 + stride_, np);
        leader_ = (rank == g0);
        first_ = offsets[static_cast<std::size_t>(g0)];
        last_ = offsets[static_cast<std::size_t>(g1)];
        active_ = leader_ && last_ > first_; // PARDISO needs >= 1 row per process
        MPI_Comm_split(comm_, group, rank, &group_comm_);
        MPI_Comm_split(comm_, active_ ? 0 : MPI_UNDEFINED, rank, &solver_comm_);
        gcnt_.assign(static_cast<std::size_t>(g1 - g0), 0);
        for (int p = g0; p < g1; ++p)
            gcnt_[static_cast<std::size_t>(p - g0)] =
                static_cast<int>(offsets[static_cast<std::size_t>(p) + 1] - offsets[static_cast<std::size_t>(p)]);
        gdsp_ = core::displacements(gcnt_);
        nprocs_ = core::allreduceSum(active_ ? 1 : 0, comm_);
#ifdef OPEN_MPI
        symmetric = false; // see the class documentation
#endif
        mtype_ = symmetric ? -2 : 11;

        // ---- send every coefficient to the leader owning its row.
        filter(entries);
        scnt_.assign(static_cast<std::size_t>(np), 0);
        rcnt_.assign(static_cast<std::size_t>(np), 0);
        std::vector<int> dest(entries.size());
        {
            utils::ScopedTimer t("setup.coarse_factorization.exchange");
            for (std::size_t k = 0; k < entries.size(); ++k) {
                const auto owner = std::upper_bound(offsets.begin(), offsets.end(), entries[k].row) - offsets.begin() - 1;
                dest[k] = static_cast<int>(owner / stride_) * stride_;
                ++scnt_[static_cast<std::size_t>(dest[k])];
            }
            MPI_Alltoall(scnt_.data(), 1, MPI_INT, rcnt_.data(), 1, MPI_INT, comm_);
            sdsp_ = core::displacements(scnt_);
            rdsp_ = core::displacements(rcnt_);
            std::vector<int> pos = sdsp_;
            nrecv_ = static_cast<std::size_t>(rdsp_.back() + rcnt_.back());
            send_pos_.resize(entries.size());
            for (std::size_t k = 0; k < entries.size(); ++k) send_pos_[k] = pos[static_cast<std::size_t>(dest[k])]++;
            std::vector<core::GlobalIndex> sr(entries.size()), sc(entries.size()), rr(nrecv_), rc(nrecv_);
            std::vector<core::Scalar> sv(entries.size()), rv(nrecv_);
            for (std::size_t k = 0; k < entries.size(); ++k) {
                const auto q = static_cast<std::size_t>(send_pos_[k]);
                sr[q] = entries[k].row;
                sc[q] = entries[k].col;
                sv[q] = entries[k].val;
            }
            const MPI_Datatype gi = core::mpiType<core::GlobalIndex>();
            MPI_Alltoallv(sr.data(), scnt_.data(), sdsp_.data(), gi, rr.data(), rcnt_.data(), rdsp_.data(), gi, comm_);
            MPI_Alltoallv(sc.data(), scnt_.data(), sdsp_.data(), gi, rc.data(), rcnt_.data(), rdsp_.data(), gi, comm_);
            MPI_Alltoallv(sv.data(), scnt_.data(), sdsp_.data(), MPI_DOUBLE, rv.data(), rcnt_.data(), rdsp_.data(),
                          MPI_DOUBLE, comm_);
            if (active_) {
                buildPattern(rr, rc);
                fillValues(rv);
            }
        }

        // ---- analysis + numerical factorization (collective over the active leaders).
        MKL_INT error = 0;
        if (!active_) { // same timer list on every rank (the report follows rank 0)
            utils::TimerRegistry::instance().add("setup.coarse_factorization.analysis", 0.0);
            utils::TimerRegistry::instance().add("setup.coarse_factorization.numeric", 0.0);
        }
        if (active_) {
            fcomm_ = static_cast<int>(MPI_Comm_c2f(solver_comm_));
            std::fill(std::begin(pt_), std::end(pt_), nullptr);
            std::fill(std::begin(iparm_), std::end(iparm_), 0);
            iparm_[0] = 1;                               // no solver defaults
            iparm_[1] = 2;                               // METIS nested dissection
            iparm_[7] = 0;                               // no refinement (2 automatic steps if pivots perturbed)
            iparm_[9] = symmetric ? 8 : 13;              // pivot perturbation 1e-8 / 1e-13
            iparm_[10] = symmetric ? 0 : 1;              // scaling
            iparm_[12] = symmetric ? 0 : 1;              // weighted matching
            iparm_[17] = -1;                             // report nnz of the factors
            iparm_[20] = 1;                              // Bunch-Kaufman pivoting (mtype -2)
            iparm_[34] = 0;                              // one-based ia/ja and domain bounds
            iparm_[39] = 2;                              // distributed A, rhs and solution
            iparm_[40] = static_cast<MKL_INT>(first_ + 1);
            iparm_[41] = static_cast<MKL_INT>(last_);
            {
                utils::ScopedTimer t("setup.coarse_factorization.analysis");
                MKL_INT phase = 11;
                error = call(phase, nullptr, nullptr);
            }
            if (error == 0) {
                utils::ScopedTimer t("setup.coarse_factorization.numeric");
                MKL_INT phase = 22;
                error = call(phase, nullptr, nullptr);
            }
            factorized_ = true; // memory to release even after an error
            factor_nnz_ = static_cast<long long>(iparm_[17]);
            perturbed_ = static_cast<long long>(iparm_[13]);
        }
        const auto err = static_cast<long long>(core::allreduceMin(static_cast<core::GlobalIndex>(error), comm_));
        factor_nnz_ = core::allreduceMax(static_cast<core::GlobalIndex>(factor_nnz_), comm_);
        perturbed_ = core::allreduceMax(static_cast<core::GlobalIndex>(perturbed_), comm_);
        if (err != 0)
            throw std::runtime_error("MKL cluster_sparse_solver failed on the coarse matrix, error = " +
                                     std::to_string(err));
        if (active_) {
            rhs_.resize(static_cast<std::size_t>(last_ - first_));
            sol_.resize(static_cast<std::size_t>(last_ - first_));
        }
    }

    void update(std::vector<core::GlobalTriplet>& entries) override
    {
        filter(entries);
        if (entries.size() != send_pos_.size()) throw std::logic_error("cpardiso coarse solver: the coarse pattern changed");
        // ---- only the values travel, along the communication pattern of setup().
        std::vector<core::Scalar> sv(entries.size()), rv(nrecv_);
        for (std::size_t k = 0; k < entries.size(); ++k) sv[static_cast<std::size_t>(send_pos_[k])] = entries[k].val;
        MPI_Alltoallv(sv.data(), scnt_.data(), sdsp_.data(), MPI_DOUBLE, rv.data(), rcnt_.data(), rdsp_.data(),
                      MPI_DOUBLE, comm_);
        // ---- phase 22 with the analysis of the last full factorization.
        MKL_INT error = 0;
        long long perturbed = 0;
        if (active_) {
            fillValues(rv);
            MKL_INT phase = 22;
            error = call(phase, nullptr, nullptr);
            perturbed = static_cast<long long>(iparm_[13]);
        }
        long long err = core::allreduceMin(static_cast<core::GlobalIndex>(error), comm_);
        perturbed = core::allreduceMax(static_cast<core::GlobalIndex>(perturbed), comm_);
        if (err != 0 || perturbed > perturbed_) { // stale scaling/matching: full analysis + factorization
            ++full_refactorizations_;
            if (active_) {
                MKL_INT phase = 11;
                error = call(phase, nullptr, nullptr);
                if (error == 0) {
                    phase = 22;
                    error = call(phase, nullptr, nullptr);
                }
                perturbed = static_cast<long long>(iparm_[13]);
            }
            err = core::allreduceMin(static_cast<core::GlobalIndex>(error), comm_);
            perturbed_ = core::allreduceMax(static_cast<core::GlobalIndex>(perturbed), comm_);
        }
        if (err != 0)
            throw std::runtime_error("MKL cluster_sparse_solver: coarse refactorization failed, error = " +
                                     std::to_string(err));
    }

    void solve(const core::Vec& c, core::Vec& y) const override
    {
        MPI_Gatherv(c.data(), static_cast<int>(c.size()), MPI_DOUBLE, rhs_.data(), gcnt_.data(), gdsp_.data(),
                    MPI_DOUBLE, 0, group_comm_);
        if (active_) {
            MKL_INT phase = 33;
            const MKL_INT error = call(phase, rhs_.data(), sol_.data());
            if (error != 0)
                throw std::runtime_error("MKL cluster_sparse_solver: coarse solve failed, error = " +
                                         std::to_string(error));
        }
        y.resize(c.size());
        MPI_Scatterv(sol_.data(), gcnt_.data(), gdsp_.data(), MPI_DOUBLE, y.data(), static_cast<int>(y.size()),
                     MPI_DOUBLE, 0, group_comm_);
    }

    long fallbacks() const override { return full_refactorizations_; }

    std::string name() const override
    {
        return "cpardiso(" + std::to_string(nprocs_) + " procs, mtype " + std::to_string(mtype_) + ", nnz(LU) " +
               std::to_string(factor_nnz_) + ", perturbed pivots " + std::to_string(perturbed_) +
               (full_refactorizations_ ? ", full refactorizations " + std::to_string(full_refactorizations_) : "") + ")";
    }

private:
    /// @brief mtype -2: upper triangle only (same filter in setup() and update()).
    void filter(std::vector<core::GlobalTriplet>& entries) const
    {
        if (mtype_ == -2)
            entries.erase(std::remove_if(entries.begin(), entries.end(),
                                         [](const core::GlobalTriplet& t) { return t.row > t.col; }),
                          entries.end());
    }

    /**
     * @brief Sorted, duplicate-free one-based CSR pattern of the rows [first_, last_) (diagonal
     *        always stored) and, for every received coefficient, its position in a_ (recv_pos_).
     */
    void buildPattern(const std::vector<core::GlobalIndex>& r, const std::vector<core::GlobalIndex>& c)
    {
        const auto nloc = static_cast<std::size_t>(last_ - first_);
        std::vector<std::size_t> cnt(nloc + 1, 0);
        for (const auto row : r) ++cnt[static_cast<std::size_t>(row - first_) + 1];
        for (std::size_t i = 0; i < nloc; ++i) cnt[i + 1] += cnt[i];
        std::vector<std::pair<core::GlobalIndex, std::size_t>> e(r.size()); // (column, received index)
        {
            std::vector<std::size_t> pos(cnt.begin(), cnt.end() - 1);
            for (std::size_t k = 0; k < r.size(); ++k) e[pos[static_cast<std::size_t>(r[k] - first_)]++] = {c[k], k};
        }
        ia_.assign(nloc + 1, 1);
        ja_.clear();
        recv_pos_.assign(r.size(), 0);
        for (std::size_t i = 0; i < nloc; ++i) {
            auto b = e.begin() + static_cast<std::ptrdiff_t>(cnt[i]), f = e.begin() + static_cast<std::ptrdiff_t>(cnt[i + 1]);
            std::sort(b, f, [](const auto& x, const auto& y) { return x.first < y.first; });
            const core::GlobalIndex diag = first_ + static_cast<core::GlobalIndex>(i);
            const std::size_t row_start = ja_.size();
            bool has_diag = false;
            for (auto it = b; it != f; ++it) {
                const auto col = static_cast<MKL_INT>(it->first + 1);
                if (ja_.size() > row_start && ja_.back() == col) { // duplicate: summed into the same position
                    recv_pos_[it->second] = ja_.size() - 1;
                    continue;
                }
                if (!has_diag && it->first > diag) { // diagonal kept in place (needed for mtype -2)
                    ja_.push_back(static_cast<MKL_INT>(diag + 1));
                    has_diag = true;
                }
                has_diag = has_diag || it->first == diag;
                ja_.push_back(col);
                recv_pos_[it->second] = ja_.size() - 1;
            }
            if (!has_diag) ja_.push_back(static_cast<MKL_INT>(diag + 1));
            if (ja_.size() > static_cast<std::size_t>(INT_MAX))
                throw std::runtime_error("coarse matrix block too large for 32-bit MKL PARDISO (use more --coarse-procs)");
            ia_[i + 1] = static_cast<MKL_INT>(ja_.size()) + 1;
        }
        a_.assign(ja_.size(), 0.0);
    }

    /// @brief a_ = sum of the received values at their CSR positions (inserted diagonals stay 0).
    void fillValues(const std::vector<core::Scalar>& v)
    {
        std::fill(a_.begin(), a_.end(), 0.0);
        for (std::size_t k = 0; k < v.size(); ++k) a_[recv_pos_[k]] += v[k];
    }

    /// @brief One call of cluster_sparse_solver with the stored matrix.
    MKL_INT call(MKL_INT phase, double* b, double* x) const
    {
        const MKL_INT maxfct = 1, mnum = 1, nrhs = 1, msglvl = 0;
        MKL_INT error = 0, idum = 0;
        cluster_sparse_solver(pt_, &maxfct, &mnum, &mtype_, &phase, &n0_, a_.data(), ia_.data(), ja_.data(), &idum,
                              &nrhs, iparm_, &msglvl, b, x, &fcomm_, &error);
        return error;
    }

    int coarse_procs_;
    int stride_ = 1, nprocs_ = 0, fcomm_ = 0;
    bool leader_ = false, active_ = false, factorized_ = false;
    core::GlobalIndex first_ = 0, last_ = 0;
    MPI_Comm comm_ = MPI_COMM_NULL, group_comm_ = MPI_COMM_NULL, solver_comm_ = MPI_COMM_NULL;
    std::vector<int> gcnt_, gdsp_;
    std::vector<int> scnt_, sdsp_, rcnt_, rdsp_;  ///< Alltoallv pattern of the coefficients (kept for update()).
    std::vector<int> send_pos_;                   ///< Coefficient k -> position in the send buffer.
    std::size_t nrecv_ = 0;                       ///< Coefficients received (leaders).
    std::vector<std::size_t> recv_pos_;           ///< Received coefficient -> position in a_ (active leaders).
    std::vector<MKL_INT> ia_, ja_;
    std::vector<core::Scalar> a_;
    MKL_INT n0_ = 0, mtype_ = 11;
    long long factor_nnz_ = 0, perturbed_ = 0;
    int full_refactorizations_ = 0;               ///< update() calls that needed phases 11 + 22.
    mutable void* pt_[64] = {};
    mutable MKL_INT iparm_[64] = {};
    mutable std::vector<core::Scalar> rhs_, sol_;
};
#endif

std::vector<std::string> availableCoarseSolvers()
{
    std::vector<std::string> v;
#ifdef USE_MUMPS
    v.push_back("mumps");
#endif
#ifdef USE_CPARDISO
    v.push_back("cpardiso");
#endif
    v.push_back("root");
    return v;
}

std::unique_ptr<CoarseSolver> makeCoarseSolver(const std::string& name, int coarse_procs,
                                               const std::string& local_solver)
{
    (void)coarse_procs; // unused when no distributed coarse solver is compiled in
    const std::string n = (name == "auto") ? availableCoarseSolvers().front() : name;
#ifdef USE_MUMPS
    if (n == "mumps") return std::make_unique<MumpsCoarseSolver>(coarse_procs);
#endif
#ifdef USE_CPARDISO
    if (n == "cpardiso") return std::make_unique<CPardisoCoarseSolver>(coarse_procs);
#endif
    if (n == "root") return std::make_unique<RootCoarseSolver>(local_solver);
    throw std::invalid_argument("coarse solver '" + n + "' is unknown or not compiled in (USE_MUMPS / USE_CPARDISO)");
}

} // namespace schwarz2lvl
