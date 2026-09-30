/**
 * @file coarse_space.cpp
 * @brief Implementation of CoarseSpace (local basis, coarse numbering, A_00 assembly).
 */
#include "coarse_space.hpp"

#include "mpi_utils.hpp"
#include "timer.hpp"

#include <Eigen/QR>

#include <algorithm>
#include <utility>

namespace schwarz2lvl::prec {

using SparseRow = std::vector<std::pair<core::GlobalIndex, core::Scalar>>; ///< (global coarse column, value)

/// @brief Packs rows into (lengths + columns, values) buffers.
void packRows(const std::vector<SparseRow>& rows, const std::vector<core::LocalIndex>& which, std::vector<core::GlobalIndex>& ib,
              std::vector<core::Scalar>& vb)
{
    ib.clear();
    vb.clear();
    for (core::LocalIndex l : which) ib.push_back(static_cast<core::GlobalIndex>(rows[static_cast<std::size_t>(l)].size()));
    for (core::LocalIndex l : which)
        for (const auto& [c, v] : rows[static_cast<std::size_t>(l)]) {
            ib.push_back(c);
            vb.push_back(v);
        }
}

/// @brief Unpacks @p count rows; row j is appended to rows[target(j)].
template <typename Target>
void unpackRows(const std::vector<core::GlobalIndex>& ib, const std::vector<core::Scalar>& vb, core::LocalIndex count, Target target,
                std::vector<SparseRow>& rows)
{
    std::size_t pc = static_cast<std::size_t>(count), pv = 0;
    for (core::LocalIndex j = 0; j < count; ++j) {
        auto& row = rows[static_cast<std::size_t>(target(j))];
        const auto len = static_cast<std::size_t>(ib[static_cast<std::size_t>(j)]);
        for (std::size_t q = 0; q < len; ++q, ++pc, ++pv) row.emplace_back(ib[pc], vb[pv]);
    }
}

CoarseSpace::CoarseSpace(const part::Subdomain& sd, const core::Mat& Z, double orth_tol) : sd_(sd)
{
    utils::ScopedTimer timer("setup.coarse_basis");
    const core::LocalIndex nl = sd_.nLocal();
    // V_i = orth(D_i Z_i): column-pivoted QR with a relative rank threshold.
    core::Mat DZ = sd_.pou().asDiagonal() * Z;
    if (DZ.cols() > 0) {
        Eigen::ColPivHouseholderQR<core::Mat> qr(DZ);
        qr.setThreshold(orth_tol);
        const Eigen::Index r = qr.rank();
        V_ = qr.householderQ() * core::Mat::Identity(nl, r);
        if (sd_.booleanPou()) V_.bottomRows(sd_.nGhost()).setZero(); // exact zeros where D_i = 0
    } else {
        V_.resize(nl, 0);
    }

    // Coarse numbering: process i owns [offsets[i], offsets[i+1]).
    const int np = core::mpiSize(sd_.comm());
    core::GlobalIndex m = static_cast<core::GlobalIndex>(V_.cols());
    std::vector<core::GlobalIndex> all(static_cast<std::size_t>(np));
    MPI_Allgather(&m, 1, core::mpiType<core::GlobalIndex>(), all.data(), 1, core::mpiType<core::GlobalIndex>(), sd_.comm());
    offsets_.assign(static_cast<std::size_t>(np) + 1, 0);
    for (int p = 0; p < np; ++p) offsets_[static_cast<std::size_t>(p) + 1] = offsets_[static_cast<std::size_t>(p)] + all[static_cast<std::size_t>(p)];
    loc_ = core::Vec::Zero(nl);
}

void CoarseSpace::assemble(const std::string& coarse_solver, int coarse_procs, const std::string& local_solver,
                           bool symmetric)
{
    const core::LocalIndex nI = sd_.nOwned(), nl = sd_.nLocal();
    const linalg::HaloExchange& halo = sd_.halo();
    const int nn = halo.numNeighbors();
    const core::GlobalIndex off = offsets_[static_cast<std::size_t>(core::mpiRank(sd_.comm()))];
    std::vector<core::GlobalTriplet> entries;
    {
        utils::ScopedTimer timer("setup.coarse_assembly");
        // ---- 1. Owned rows of W = R_0^T: own V_i rows (+ neighbours' ghost rows of V_j).
        std::vector<SparseRow> rows(static_cast<std::size_t>(nl));
        for (core::LocalIndex l = 0; l < nI; ++l)
            for (Eigen::Index c = 0; c < V_.cols(); ++c)
                if (V_(l, c) != 0.0) rows[static_cast<std::size_t>(l)].emplace_back(off + c, V_(l, c));
        std::vector<std::vector<core::GlobalIndex>> si(static_cast<std::size_t>(nn)), ri;
        std::vector<std::vector<core::Scalar>> sv(static_cast<std::size_t>(nn)), rv;
        if (!sd_.booleanPou()) {
            // Ghost rows of V_i belong to W rows owned by the neighbours (reverse direction).
            std::vector<SparseRow> ghostRows(static_cast<std::size_t>(nl));
            for (core::LocalIndex l = nI; l < nl; ++l)
                for (Eigen::Index c = 0; c < V_.cols(); ++c)
                    if (V_(l, c) != 0.0) ghostRows[static_cast<std::size_t>(l)].emplace_back(off + c, V_(l, c));
            for (int k = 0; k < nn; ++k) {
                std::vector<core::LocalIndex> which(static_cast<std::size_t>(halo.recvCount(k)));
                for (core::LocalIndex j = 0; j < halo.recvCount(k); ++j) which[static_cast<std::size_t>(j)] = halo.recvStart(k) + j;
                packRows(ghostRows, which, si[static_cast<std::size_t>(k)], sv[static_cast<std::size_t>(k)]);
            }
            halo.exchange(si, ri);
            halo.exchange(sv, rv);
            for (int k = 0; k < nn; ++k) {
                const auto& send = halo.sendIndices(k); // my owned unknowns held by neighbour k, in its ghost order
                unpackRows(ri[static_cast<std::size_t>(k)], rv[static_cast<std::size_t>(k)],
                           static_cast<core::LocalIndex>(send.size()), [&](core::LocalIndex j) { return send[static_cast<std::size_t>(j)]; }, rows);
            }
        }

        // ---- 2. Ghost rows of W: forward exchange of the assembled owned rows.
        for (int k = 0; k < nn; ++k)
            packRows(rows, halo.sendIndices(k), si[static_cast<std::size_t>(k)], sv[static_cast<std::size_t>(k)]);
        halo.exchange(si, ri);
        halo.exchange(sv, rv);
        for (int k = 0; k < nn; ++k) {
            const core::LocalIndex start = halo.recvStart(k);
            unpackRows(ri[static_cast<std::size_t>(k)], rv[static_cast<std::size_t>(k)], halo.recvCount(k),
                       [&](core::LocalIndex j) { return start + j; }, rows);
        }

        // ---- 3. Local compressed column numbering of W(Omega_i, :).
        std::vector<core::GlobalIndex> cols;
        for (const auto& r : rows)
            for (const auto& e : r) cols.push_back(e.first);
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        auto lc = [&](core::GlobalIndex g) {
            return static_cast<core::LocalIndex>(std::lower_bound(cols.begin(), cols.end(), g) - cols.begin());
        };
        const auto nc = static_cast<core::LocalIndex>(cols.size());
        std::vector<core::Triplet> tw;
        for (core::LocalIndex l = 0; l < nl; ++l)
            for (const auto& [c, v] : rows[static_cast<std::size_t>(l)]) tw.emplace_back(l, lc(c), v);
        core::SpMat W(nl, nc);
        W.setFromTriplets(tw.begin(), tw.end());

        // ---- 4. A W restricted to the owned rows: A(Omega_I, Omega_i) W(Omega_i, :).
        const core::SpMat AW = sd_.Aowned() * W;

        // ---- 5. Local term W(Omega_I, :)^T (A W)(Omega_I, :) with a dense left factor
        //         (its nonzero columns only: m_i with the Boolean PoU).
        std::vector<core::LocalIndex> wc;
        for (core::LocalIndex l = 0; l < nI; ++l)
            for (const auto& e : rows[static_cast<std::size_t>(l)]) wc.push_back(lc(e.first));
        std::sort(wc.begin(), wc.end());
        wc.erase(std::unique(wc.begin(), wc.end()), wc.end());
        core::Mat Wd = core::Mat::Zero(nI, static_cast<Eigen::Index>(wc.size()));
        for (core::LocalIndex l = 0; l < nI; ++l)
            for (const auto& [c, v] : rows[static_cast<std::size_t>(l)]) {
                const auto pos = std::lower_bound(wc.begin(), wc.end(), lc(c)) - wc.begin();
                Wd(l, pos) += v;
            }
        const core::Mat E = Wd.transpose() * AW; // |wc| x nc
        for (Eigen::Index a = 0; a < E.rows(); ++a)
            for (Eigen::Index b = 0; b < E.cols(); ++b)
                if (E(a, b) != 0.0)
                    entries.push_back({cols[static_cast<std::size_t>(wc[static_cast<std::size_t>(a)])],
                                       cols[static_cast<std::size_t>(b)], E(a, b)});
        local_entries_ = entries.size();
    }
    utils::ScopedTimer timer("setup.coarse_factorization");
    solver_ = makeCoarseSolver(coarse_solver, coarse_procs, local_solver);
    solver_->setup(sd_.comm(), offsets_, entries, symmetric);
}

void CoarseSpace::restrict(const core::Vec& r, core::Vec& c) const
{
    const core::LocalIndex nI = sd_.nOwned();
    if (sd_.booleanPou()) {
        c.noalias() = V_.topRows(nI).transpose() * r; // V_i = 0 on the ghosts
        return;
    }
    loc_.head(nI) = r;
    sd_.halo().forward(loc_);
    c.noalias() = V_.transpose() * loc_;
}

void CoarseSpace::prolong(const core::Vec& y, core::Vec& x) const
{
    const core::LocalIndex nI = sd_.nOwned();
    if (sd_.booleanPou()) {
        x.noalias() = V_.topRows(nI) * y;
        return;
    }
    loc_.noalias() = V_ * y;
    sd_.halo().reverseAdd(loc_);
    x = loc_.head(nI);
}

void CoarseSpace::solve(const core::Vec& c, core::Vec& y) const { solver_->solve(c, y); }

void CoarseSpace::apply(const core::Vec& r, core::Vec& x) const
{
    restrict(r, c_);
    solve(c_, y_);
    prolong(y_, x);
}

} // namespace schwarz2lvl::part
