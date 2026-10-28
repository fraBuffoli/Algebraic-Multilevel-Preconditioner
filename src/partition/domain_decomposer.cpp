/**
 * @file domain_decomposer.cpp
 * @brief Implementation of DomainDecomposer (rank-0 load / METIS / scatter).
 */
#include "domain_decomposer.hpp"

#include "graph_partitioner.hpp"
#include "logger.hpp"
#include "matrix_generators.hpp"
#include "matrix_market_io.hpp"
#include "mpi_utils.hpp"
#include "timer.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace schwarz2lvl::part {

constexpr int kTagHeader = 101; ///< Tag: sizes of the owned block.
constexpr int kTagRowPtr = 102; ///< Tag: CSR row pointer.
constexpr int kTagCol = 103;    ///< Tag: CSR column indices.
constexpr int kTagVal = 104;    ///< Tag: CSR values.
constexpr int kTagRhs = 105;    ///< Tag: right-hand side.

/// @brief Checks that a message size fits in an int (MPI count).
int toCount(std::size_t n)
{
    if (n > static_cast<std::size_t>(INT_MAX))
        throw std::runtime_error("DomainDecomposer: message too large for a single MPI call");
    return static_cast<int>(n);
}

/// @brief Numerical symmetry test (|a_ij - a_ji| <= tol * max|a|), rank 0 only.
bool isNumericallySymmetric(const CsrMatrix& A)
{
    core::Scalar amax = 0.0;
    for (core::Scalar v : A.val) amax = std::max(amax, std::abs(v));
    const core::Scalar tol = 1e-14 * amax;
    for (core::GlobalIndex i = 0; i < A.nrows; ++i) {
        for (std::int64_t k = A.row_ptr[static_cast<std::size_t>(i)]; k < A.row_ptr[static_cast<std::size_t>(i) + 1]; ++k) {
            const core::GlobalIndex j = A.col[static_cast<std::size_t>(k)];
            if (j <= i) continue;
            const auto b = A.col.begin() + A.row_ptr[static_cast<std::size_t>(j)];
            const auto e = A.col.begin() + A.row_ptr[static_cast<std::size_t>(j) + 1];
            const auto it = std::lower_bound(b, e, i);
            const core::Scalar aji = (it != e && *it == i) ? A.val[static_cast<std::size_t>(it - A.col.begin())] : 0.0;
            if (std::abs(aji - A.val[static_cast<std::size_t>(k)]) > tol) return false;
        }
    }
    return true;
}

/// @brief Everything rank 0 builds before scattering.
struct GlobalProblem {
    CsrMatrix A;              ///< Original matrix.
    std::vector<core::Scalar> b;    ///< Original right-hand side.
    int block_size = 1;       ///< Unknowns per cell.
    std::string description;  ///< Description.
};

/**
 * @brief File name of step @p step: a pattern with one integer conversion (%d, %4d, %04d)
 *        is formatted with the step; a name without '%' is returned unchanged.
 */
std::string stepFileName(const std::string& pattern, int step)
{
    const auto p = pattern.find('%');
    if (p == std::string::npos) return pattern;
    std::size_t q = p + 1;
    const bool zero = q < pattern.size() && pattern[q] == '0';
    if (zero) ++q;
    int width = 0;
    while (q < pattern.size() && std::isdigit(static_cast<unsigned char>(pattern[q]))) width = 10 * width + (pattern[q++] - '0');
    if (q >= pattern.size() || pattern[q] != 'd' || pattern.find('%', q + 1) != std::string::npos)
        throw std::invalid_argument("--matrix: a step pattern must contain exactly one %d / %0Nd conversion");
    std::string num = std::to_string(step);
    if (static_cast<int>(num.size()) < width) num.insert(0, static_cast<std::size_t>(width) - num.size(), zero ? '0' : ' ');
    return pattern.substr(0, p) + num + pattern.substr(q + 1);
}

/// @brief Generator parameters of the configuration at pseudo-time @p step.
utils::GeneratorOptions generatorOptions(const core::SolverConfig& cfg, int step)
{
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
    o.source = cfg.bubble_source;
    return o;
}

/// @brief FNV-1a hash of the sparsity pattern (size, row pointer, columns).
std::uint64_t hashPattern(const CsrMatrix& A)
{
    std::uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](std::uint64_t v) {
        for (int b = 0; b < 8; ++b) {
            h ^= (v >> (8 * b)) & 0xffULL;
            h *= 1099511628211ULL;
        }
    };
    mix(static_cast<std::uint64_t>(A.nrows));
    for (const auto v : A.row_ptr) mix(static_cast<std::uint64_t>(v));
    for (const auto v : A.col) mix(static_cast<std::uint64_t>(v));
    return h;
}

/// @brief Rank 0: reads or generates the matrix of step @p step and builds the right-hand side.
GlobalProblem loadProblem(const core::SolverConfig& cfg, int step = 0)
{
    GlobalProblem g;
    std::vector<core::Scalar> generated_rhs;
    if (!cfg.generate.empty()) {
        utils::ScopedTimer t("decomposition.generate");
        utils::GeneratedProblem p = utils::MatrixGenerator::generate(cfg.generate, generatorOptions(cfg, step));
        g.A = std::move(p.A);
        generated_rhs = std::move(p.rhs);
        g.block_size = p.block_size;
        g.description = p.description;
    } else {
        utils::ScopedTimer t("decomposition.read");
        const std::string file = stepFileName(cfg.matrix_file, step);
        g.A = MatrixMarketIO::readMatrix(file);
        std::ostringstream d;
        d << file << " (n=" << g.A.nrows << ", nnz=" << g.A.nnz() << ")";
        g.description = d.str();
    }
    if (g.A.nrows != g.A.ncols) throw std::runtime_error("the matrix must be square");
    if (cfg.block_size > 0) g.block_size = cfg.block_size;
    if (g.A.nrows % g.block_size != 0) throw std::runtime_error("n is not a multiple of the block size");
    if (!g.A.symmetric) g.A.symmetric = isNumericallySymmetric(g.A);

    const std::size_t n = static_cast<std::size_t>(g.A.nrows);
    if (cfg.rhs == "random") {
        // Random right-hand side as in Section 4 of the paper; generated in the
        // original ordering so that it does not depend on the number of processes.
        std::mt19937_64 gen(cfg.seed);
        std::uniform_real_distribution<core::Scalar> dist(-1.0, 1.0);
        g.b.resize(n);
        for (auto& v : g.b) v = dist(gen);
    } else if (cfg.rhs == "ones") {
        g.b.assign(n, 1.0);
    } else if (cfg.rhs == "generated") {
        if (generated_rhs.empty()) throw std::runtime_error("--rhs generated requires --generate");
        g.b = std::move(generated_rhs);
    } else {
        g.b = MatrixMarketIO::readVector(cfg.rhs);
        if (g.b.size() != n) throw std::runtime_error("right-hand side size does not match the matrix");
    }
    return g;
}

/// @brief Extracts the rows [begin, end) of the permuted matrix as CSR with permuted columns.
void extractRows(const CsrMatrix& A, const std::vector<core::Scalar>& b, const std::vector<core::GlobalIndex>& new_to_old,
                 const std::vector<core::GlobalIndex>& old_to_new, core::GlobalIndex begin, core::GlobalIndex end,
                 std::vector<core::GlobalIndex>& row_ptr, std::vector<core::GlobalIndex>& col, std::vector<core::Scalar>& val,
                 std::vector<core::Scalar>& rhs)
{
    const std::size_t nr = static_cast<std::size_t>(end - begin);
    row_ptr.assign(nr + 1, 0);
    std::int64_t nnz = 0;
    for (std::size_t r = 0; r < nr; ++r) {
        const auto old = static_cast<std::size_t>(new_to_old[static_cast<std::size_t>(begin) + r]);
        nnz += A.row_ptr[old + 1] - A.row_ptr[old];
        row_ptr[r + 1] = nnz;
    }
    col.resize(static_cast<std::size_t>(nnz));
    val.resize(static_cast<std::size_t>(nnz));
    rhs.resize(nr);
    std::vector<std::pair<core::GlobalIndex, core::Scalar>> row;
    for (std::size_t r = 0; r < nr; ++r) {
        const auto old = static_cast<std::size_t>(new_to_old[static_cast<std::size_t>(begin) + r]);
        row.clear();
        for (std::int64_t k = A.row_ptr[old]; k < A.row_ptr[old + 1]; ++k)
            row.emplace_back(old_to_new[static_cast<std::size_t>(A.col[static_cast<std::size_t>(k)])],
                             A.val[static_cast<std::size_t>(k)]);
        std::sort(row.begin(), row.end());
        std::int64_t pos = row_ptr[r];
        for (const auto& [c, v] : row) {
            col[static_cast<std::size_t>(pos)] = c;
            val[static_cast<std::size_t>(pos)] = v;
            ++pos;
        }
        rhs[r] = b[old];
    }
}

/// @brief Broadcasts a string from rank 0.
void bcastString(std::string& s, MPI_Comm comm)
{
    long long len = static_cast<long long>(s.size());
    MPI_Bcast(&len, 1, MPI_LONG_LONG, 0, comm);
    s.resize(static_cast<std::size_t>(len));
    if (len > 0) MPI_Bcast(s.data(), static_cast<int>(len), MPI_CHAR, 0, comm);
}

core::LocalSystem DomainDecomposer::decompose(const core::SolverConfig& cfg, MPI_Comm comm, DecompositionInfo& info)
{
    utils::ScopedTimer timer("decomposition");
    const int rank = core::mpiRank(comm);
    const int nranks = core::mpiSize(comm);

    core::LocalSystem sys;
    sys.comm = comm;
    sys.ownership.assign(static_cast<std::size_t>(nranks) + 1, 0);

    // ------------------------------------------------------------------ rank 0
    GlobalProblem g;
    std::vector<core::GlobalIndex> new_to_old, old_to_new;
    std::string error;
    long long meta[4] = {0, 0, 0, 0}; // n, nnz, block_size, symmetric
    if (rank == 0) {
        try {
            g = loadProblem(cfg);
            const std::size_t n = static_cast<std::size_t>(g.A.nrows);

            // METIS partition of G(A + A^T): one subdomain per process.
            std::vector<int> part(n, 0);
            if (nranks > 1) {
                utils::ScopedTimer t("decomposition.metis");
                GraphPartitioner::Options opt;
                opt.objective = cfg.metis_objective;
                opt.vertex_weights = cfg.metis_vertex_weights;
                opt.block_size = g.block_size;
                part = GraphPartitioner::partition(g.A, nranks, opt);
            }

            // Renumbering: unknowns of part p get the contiguous range
            // [ownership[p], ownership[p+1]); inside a part the original order is kept.
            std::vector<core::GlobalIndex> count(static_cast<std::size_t>(nranks), 0);
            for (int p : part) ++count[static_cast<std::size_t>(p)];
            for (int p = 0; p < nranks; ++p)
                sys.ownership[static_cast<std::size_t>(p) + 1] = sys.ownership[static_cast<std::size_t>(p)] + count[static_cast<std::size_t>(p)];
            for (int p = 0; p < nranks; ++p)
                if (count[static_cast<std::size_t>(p)] == 0) throw std::runtime_error("METIS produced an empty subdomain");
            new_to_old.resize(n);
            old_to_new.resize(n);
            std::vector<core::GlobalIndex> next(sys.ownership.begin(), sys.ownership.end() - 1);
            for (std::size_t i = 0; i < n; ++i) {
                const core::GlobalIndex k = next[static_cast<std::size_t>(part[i])]++;
                new_to_old[static_cast<std::size_t>(k)] = static_cast<core::GlobalIndex>(i);
                old_to_new[i] = k;
            }
            info.pattern_hash = hashPattern(g.A);
            meta[0] = static_cast<long long>(g.A.nrows);
            meta[1] = static_cast<long long>(g.A.nnz());
            meta[2] = g.block_size;
            meta[3] = g.A.symmetric ? 1 : 0;
            info.part = std::move(part);
            info.description = g.description;
        } catch (const std::exception& e) {
            error = e.what();
        }
    }
    // Errors on rank 0 are propagated to everybody (no deadlock).
    bcastString(error, comm);
    if (!error.empty()) throw std::runtime_error(error);

    MPI_Bcast(meta, 4, MPI_LONG_LONG, 0, comm);
    MPI_Bcast(sys.ownership.data(), nranks + 1, core::mpiType<core::GlobalIndex>(), 0, comm);
    bcastString(info.description, comm);
    sys.n_global = static_cast<core::GlobalIndex>(meta[0]);
    info.nnz = static_cast<std::int64_t>(meta[1]);
    sys.block_size = static_cast<int>(meta[2]);
    sys.symmetric = (meta[3] != 0);

    // ---------------------------------------------------------------- scatter
    utils::ScopedTimer tscatter("decomposition.scatter");
    if (rank == 0) {
        std::vector<std::int64_t> rp;
        std::vector<core::GlobalIndex> cl;
        std::vector<core::Scalar> vl, rh;
        for (int p = 1; p < nranks; ++p) {
            extractRows(g.A, g.b, new_to_old, old_to_new, sys.ownership[static_cast<std::size_t>(p)],
                        sys.ownership[static_cast<std::size_t>(p) + 1], rp, cl, vl, rh);
            long long hdr[2] = {static_cast<long long>(rh.size()), static_cast<long long>(cl.size())};
            MPI_Send(hdr, 2, MPI_LONG_LONG, p, kTagHeader, comm);
            MPI_Send(rp.data(), toCount(rp.size()), core::mpiType<core::GlobalIndex>(), p, kTagRowPtr, comm);
            MPI_Send(cl.data(), toCount(cl.size()), core::mpiType<core::GlobalIndex>(), p, kTagCol, comm);
            MPI_Send(vl.data(), toCount(vl.size()), MPI_DOUBLE, p, kTagVal, comm);
            MPI_Send(rh.data(), toCount(rh.size()), MPI_DOUBLE, p, kTagRhs, comm);
        }
        extractRows(g.A, g.b, new_to_old, old_to_new, sys.ownership[0], sys.ownership[1], sys.row_ptr, sys.col,
                    sys.val, sys.rhs);
        if (info.keep_global) {
            // Permuted copy P A P^T and P b for the tests.
            extractRows(g.A, g.b, new_to_old, old_to_new, 0, g.A.nrows, info.A_perm.row_ptr, info.A_perm.col,
                        info.A_perm.val, info.b_perm);
            info.A_perm.nrows = info.A_perm.ncols = g.A.nrows;
            info.A_perm.symmetric = g.A.symmetric;
        }
        info.new_to_old = std::move(new_to_old);
    } else {
        long long hdr[2] = {0, 0};
        MPI_Recv(hdr, 2, MPI_LONG_LONG, 0, kTagHeader, comm, MPI_STATUS_IGNORE);
        sys.row_ptr.resize(static_cast<std::size_t>(hdr[0]) + 1);
        sys.col.resize(static_cast<std::size_t>(hdr[1]));
        sys.val.resize(static_cast<std::size_t>(hdr[1]));
        sys.rhs.resize(static_cast<std::size_t>(hdr[0]));
        MPI_Recv(sys.row_ptr.data(), toCount(sys.row_ptr.size()), core::mpiType<core::GlobalIndex>(), 0, kTagRowPtr, comm,
                 MPI_STATUS_IGNORE);
        MPI_Recv(sys.col.data(), toCount(sys.col.size()), core::mpiType<core::GlobalIndex>(), 0, kTagCol, comm, MPI_STATUS_IGNORE);
        MPI_Recv(sys.val.data(), toCount(sys.val.size()), MPI_DOUBLE, 0, kTagVal, comm, MPI_STATUS_IGNORE);
        MPI_Recv(sys.rhs.data(), toCount(sys.rhs.size()), MPI_DOUBLE, 0, kTagRhs, comm, MPI_STATUS_IGNORE);
    }
    return sys;
}

void DomainDecomposer::updateValues(const core::SolverConfig& cfg, int step, core::LocalSystem& sys, DecompositionInfo& info)
{
    utils::ScopedTimer timer("decomposition.update");
    MPI_Comm comm = sys.comm;
    const int rank = core::mpiRank(comm);
    const int nranks = core::mpiSize(comm);
    GlobalProblem g;
    std::vector<core::GlobalIndex> old_to_new;
    std::string error;
    if (rank == 0) {
        try {
            g = loadProblem(cfg, step);
            if (g.A.nrows != sys.n_global || g.block_size != sys.block_size)
                throw std::runtime_error("step " + std::to_string(step) + ": size or block size changed");
            if (hashPattern(g.A) != info.pattern_hash)
                throw std::runtime_error("step " + std::to_string(step) + ": the sparsity pattern changed");
            if (g.A.symmetric != sys.symmetric)
                throw std::runtime_error("step " + std::to_string(step) + ": the symmetry of the matrix changed");
            old_to_new.resize(info.new_to_old.size());
            for (std::size_t k = 0; k < info.new_to_old.size(); ++k)
                old_to_new[static_cast<std::size_t>(info.new_to_old[k])] = static_cast<core::GlobalIndex>(k);
            info.description = g.description;
        } catch (const std::exception& e) {
            error = e.what();
        }
    }
    bcastString(error, comm);
    if (!error.empty()) throw std::runtime_error(error);
    bcastString(info.description, comm);

    // Values and right-hand side only: the receivers already have row_ptr and col, and
    // extractRows() orders every row by new column index exactly as decompose() did.
    if (rank == 0) {
        std::vector<std::int64_t> rp;
        std::vector<core::GlobalIndex> cl;
        std::vector<core::Scalar> vl, rh;
        for (int p = 1; p < nranks; ++p) {
            extractRows(g.A, g.b, info.new_to_old, old_to_new, sys.ownership[static_cast<std::size_t>(p)],
                        sys.ownership[static_cast<std::size_t>(p) + 1], rp, cl, vl, rh);
            MPI_Send(vl.data(), toCount(vl.size()), MPI_DOUBLE, p, kTagVal, comm);
            MPI_Send(rh.data(), toCount(rh.size()), MPI_DOUBLE, p, kTagRhs, comm);
        }
        extractRows(g.A, g.b, info.new_to_old, old_to_new, sys.ownership[0], sys.ownership[1], rp, cl, vl, rh);
        if (cl != sys.col) throw std::logic_error("DomainDecomposer::updateValues: column order changed");
        sys.val = std::move(vl);
        sys.rhs = std::move(rh);
        if (info.keep_global)
            extractRows(g.A, g.b, info.new_to_old, old_to_new, 0, g.A.nrows, info.A_perm.row_ptr, info.A_perm.col,
                        info.A_perm.val, info.b_perm);
    } else {
        MPI_Recv(sys.val.data(), toCount(sys.val.size()), MPI_DOUBLE, 0, kTagVal, comm, MPI_STATUS_IGNORE);
        MPI_Recv(sys.rhs.data(), toCount(sys.rhs.size()), MPI_DOUBLE, 0, kTagRhs, comm, MPI_STATUS_IGNORE);
    }
}

core::Vec DomainDecomposer::scatterOriginalOrder(const core::LocalSystem& sys, const std::vector<core::Scalar>& xg,
                                                 const DecompositionInfo& info)
{
    const int rank = core::mpiRank(sys.comm);
    const int nranks = core::mpiSize(sys.comm);
    std::vector<int> counts(static_cast<std::size_t>(nranks));
    for (int p = 0; p < nranks; ++p)
        counts[static_cast<std::size_t>(p)] = toCount(static_cast<std::size_t>(sys.ownership[static_cast<std::size_t>(p) + 1] - sys.ownership[static_cast<std::size_t>(p)]));
    const std::vector<int> displs = core::displacements(counts);
    std::vector<core::Scalar> xp;
    if (rank == 0) {
        if (xg.size() != static_cast<std::size_t>(sys.n_global)) throw std::invalid_argument("scatterOriginalOrder: wrong size");
        xp.resize(xg.size());
        for (std::size_t k = 0; k < xp.size(); ++k) xp[k] = xg[static_cast<std::size_t>(info.new_to_old[k])];
    }
    core::Vec x(sys.nOwned());
    MPI_Scatterv(xp.data(), counts.data(), displs.data(), MPI_DOUBLE, x.data(), static_cast<int>(x.size()), MPI_DOUBLE, 0,
                 sys.comm);
    return x;
}

std::vector<core::Scalar> DomainDecomposer::gather(const core::LocalSystem& sys, const core::Vec& x_owned)
{
    const int rank = core::mpiRank(sys.comm);
    const int nranks = core::mpiSize(sys.comm);
    std::vector<int> counts(static_cast<std::size_t>(nranks));
    for (int p = 0; p < nranks; ++p)
        counts[static_cast<std::size_t>(p)] = toCount(static_cast<std::size_t>(sys.ownership[static_cast<std::size_t>(p) + 1] - sys.ownership[static_cast<std::size_t>(p)]));
    const std::vector<int> displs = core::displacements(counts);
    std::vector<core::Scalar> x;
    if (rank == 0) x.resize(static_cast<std::size_t>(sys.n_global));
    MPI_Gatherv(x_owned.data(), static_cast<int>(x_owned.size()), MPI_DOUBLE, x.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, sys.comm);
    return x;
}

std::vector<core::Scalar> DomainDecomposer::gatherOriginalOrder(const core::LocalSystem& sys, const core::Vec& x_owned,
                                                          const DecompositionInfo& info)
{
    std::vector<core::Scalar> xp = gather(sys, x_owned);
    if (core::mpiRank(sys.comm) != 0) return {};
    std::vector<core::Scalar> x(xp.size());
    for (std::size_t k = 0; k < xp.size(); ++k) x[static_cast<std::size_t>(info.new_to_old[k])] = xp[k];
    return x;
}

} // namespace schwarz2lvl