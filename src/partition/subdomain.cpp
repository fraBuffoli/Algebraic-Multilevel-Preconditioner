/**
 * @file subdomain.cpp
 * @brief Implementation of Subdomain (overlap discovery, halo, local matrices, PoU).
 */
#include "subdomain.hpp"

#include "mpi_utils.hpp"
#include "timer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace schwarz2lvl::part {

namespace {
/**
 * @brief Position of the stored entry (outer, inner) in the value array of a compressed
 *        Eigen sparse matrix (outer = column for column-major, row for row-major).
 * @throws std::logic_error if the entry is not in the pattern.
 */
template <typename SparseMatrix>
core::LocalIndex storagePosition(const SparseMatrix& M, core::LocalIndex outer, core::LocalIndex inner)
{
    const core::LocalIndex* idx = M.innerIndexPtr();
    const core::LocalIndex* b = idx + M.outerIndexPtr()[outer];
    const core::LocalIndex* e = idx + M.outerIndexPtr()[outer + 1];
    const core::LocalIndex* it = std::lower_bound(b, e, inner);
    if (it == e || *it != inner) throw std::logic_error("Subdomain: entry not found in the sparsity pattern");
    return static_cast<core::LocalIndex>(it - idx);
}
} // namespace

Subdomain::Subdomain(const core::LocalSystem& sys, const std::string& pou) : sys_(sys)
{
    utils::ScopedTimer timer("setup.subdomain");
    n_owned_ = sys_.nOwned();
    begin_ = sys_.rowBegin();

    buildOverlap();

    //Fetch the ghost rows A(Omega_Gamma_i, :) from their owners.
    //For every requested unknown the owner sends [row length, columns...] and the values.
    const int nn = halo_.numNeighbors();
    std::vector<std::vector<core::GlobalIndex>> scols(static_cast<std::size_t>(nn)), rcols;
    std::vector<std::vector<core::Scalar>> rvals;
    for (int k = 0; k < nn; ++k) {
        for (core::LocalIndex l : halo_.sendIndices(k)) {
            const core::GlobalIndex b = sys_.row_ptr[static_cast<std::size_t>(l)];
            const core::GlobalIndex e = sys_.row_ptr[static_cast<std::size_t>(l) + 1];
            scols[static_cast<std::size_t>(k)].push_back(e - b);
            for (core::GlobalIndex q = b; q < e; ++q)
                scols[static_cast<std::size_t>(k)].push_back(sys_.col[static_cast<std::size_t>(q)]);
        }
    }
    halo_.exchange(scols, rcols);
    halo_.exchange(packGhostRowValues(), rvals); // same order as the columns: updateValues() relies on it

    buildMatrices(rcols, rvals);
    buildValueMaps(rcols);
    buildPartitionOfUnity(pou);
    val_snapshot_ = sys_.val;
}

std::vector<std::vector<core::Scalar>> Subdomain::packGhostRowValues() const
{
    const int nn = halo_.numNeighbors();
    std::vector<std::vector<core::Scalar>> svals(static_cast<std::size_t>(nn));
    for (int k = 0; k < nn; ++k) {
        auto& s = svals[static_cast<std::size_t>(k)];
        for (core::LocalIndex l : halo_.sendIndices(k)) {
            const auto b = static_cast<std::size_t>(sys_.row_ptr[static_cast<std::size_t>(l)]);
            const auto e = static_cast<std::size_t>(sys_.row_ptr[static_cast<std::size_t>(l) + 1]);
            s.insert(s.end(), sys_.val.begin() + static_cast<std::ptrdiff_t>(b), sys_.val.begin() + static_cast<std::ptrdiff_t>(e));
        }
    }
    return svals;
}

void Subdomain::buildValueMaps(const std::vector<std::vector<std::int64_t>>& rcols)
{
    // Owned rows: entry k of the CSR row l goes to A_ii(l, c) (column-major) and A(Omega_I, Omega_i)(l, c) (row-major).
    own_to_aii_.resize(sys_.col.size());
    own_to_aown_.resize(sys_.col.size());
    for (core::LocalIndex l = 0; l < n_owned_; ++l)
        for (core::GlobalIndex k = sys_.row_ptr[static_cast<std::size_t>(l)]; k < sys_.row_ptr[static_cast<std::size_t>(l) + 1]; ++k) {
            const core::LocalIndex c = localIndex(sys_.col[static_cast<std::size_t>(k)]);
            own_to_aii_[static_cast<std::size_t>(k)] = storagePosition(Aii_, c, l);
            own_to_aown_[static_cast<std::size_t>(k)] = storagePosition(Aowned_, l, c);
        }
    // Ghost rows: same traversal as buildMatrices().
    ghost_map_.clear();
    for (int k = 0; k < halo_.numNeighbors(); ++k) {
        const auto& cols = rcols[static_cast<std::size_t>(k)];
        std::size_t pc = 0;
        for (core::LocalIndex j = 0; j < halo_.recvCount(k); ++j) {
            const core::LocalIndex row = halo_.recvStart(k) + j;
            const core::GlobalIndex len = cols[pc++];
            for (core::GlobalIndex q = 0; q < len; ++q, ++pc) {
                const core::LocalIndex c = localIndex(cols[pc]);
                ghost_map_.push_back(c >= 0 ? storagePosition(Aii_, c, row) : -(1 + (row - n_owned_)));
            }
        }
    }
}

void Subdomain::refillValues(const std::vector<std::vector<core::Scalar>>& rvals)
{
    // Every stored entry of A_ii and A(Omega_I, Omega_i) comes from exactly one source value
    // (the matrices were assembled without duplicates), so plain assignment is enough.
    core::Scalar* aii = Aii_.valuePtr();
    core::Scalar* aown = Aowned_.valuePtr();
    for (std::size_t k = 0; k < sys_.val.size(); ++k) {
        aii[own_to_aii_[k]] = sys_.val[k];
        aown[own_to_aown_[k]] = sys_.val[k];
    }
    lump_.setZero();
    std::size_t q = 0;
    for (const auto& r : rvals)
        for (const core::Scalar v : r) {
            const core::LocalIndex m = ghost_map_[q++];
            if (m >= 0)
                aii[m] = v;
            else
                lump_[-m - 1] += std::abs(v);
        }
}

bool Subdomain::updateValues()
{
    utils::ScopedTimer timer("update.subdomain");
    if (sys_.val.size() != val_snapshot_.size())
        throw std::logic_error("Subdomain::updateValues: the number of nonzeros changed (pattern must be kept)");
    // A neighbour may have changed rows that this rank holds as ghosts: decide together.
    const bool mine = (sys_.val != val_snapshot_);
    if (core::allreduceMax(mine ? 1 : 0, sys_.comm) == 0) return false;

    std::vector<std::vector<core::Scalar>> rvals;
    halo_.exchange(packGhostRowValues(), rvals);
    std::size_t nrecv = 0;
    for (const auto& r : rvals) nrecv += r.size();
    if (nrecv != ghost_map_.size())
        throw std::logic_error("Subdomain::updateValues: ghost rows changed size (pattern must be kept)");
    refillValues(rvals);
    val_snapshot_ = sys_.val;
    ++version_;
    return true;
}

core::GlobalIndex Subdomain::globalId(core::LocalIndex l) const
{
    return (l < n_owned_) ? begin_ + l : ghosts_[static_cast<std::size_t>(l - n_owned_)];
}

core::LocalIndex Subdomain::localIndex(core::GlobalIndex g) const
{
    if (g >= begin_ && g < begin_ + n_owned_) return static_cast<core::LocalIndex>(g - begin_);
    const auto it = std::lower_bound(ghosts_.begin(), ghosts_.end(), g);
    if (it != ghosts_.end() && *it == g) return n_owned_ + static_cast<core::LocalIndex>(it - ghosts_.begin());
    return -1;
}

void Subdomain::buildOverlap()
{
    MPI_Comm comm = sys_.comm;
    const int nranks = core::mpiSize(comm);
    const core::GlobalIndex end = begin_ + n_owned_;

    // A_ij : j is a colomn of my rows, I don't need other subdomains.
    // Remember, always dealing with non-oriented graph G(A+A^T)
    std::vector<core::GlobalIndex> cand;  
    std::vector<std::pair<int, core::GlobalIndex>> notify;  
    for (core::LocalIndex l = 0; l < n_owned_; ++l) {
        for (core::GlobalIndex k = sys_.row_ptr[static_cast<std::size_t>(l)]; k < sys_.row_ptr[static_cast<std::size_t>(l) + 1]; ++k) {
            const core::GlobalIndex g = sys_.col[static_cast<std::size_t>(k)];
            if (g >= begin_ && g < end) continue; // discard internal columns
            cand.push_back(g);
            notify.emplace_back(core::ownerOf(sys_.ownership, g), begin_ + l);
        }
    }
    std::sort(notify.begin(), notify.end());
    notify.erase(std::unique(notify.begin(), notify.end()), notify.end());

    // A_ji : i is a column owned by row j of another subdomain! 
    std::vector<int> scount(static_cast<std::size_t>(nranks), 0), rcount(static_cast<std::size_t>(nranks), 0);
    for (const auto& pr : notify) ++scount[static_cast<std::size_t>(pr.first)]; //scount[q] = number of notifications sent to q
    MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, comm); //rcount[q] = number of notifications FROM q
    const std::vector<int> sdisp = core::displacements(scount), rdisp = core::displacements(rcount);
    std::vector<core::GlobalIndex> sbuf(notify.size());
    for (std::size_t k = 0; k < notify.size(); ++k) sbuf[k] = notify[k].second;
    std::vector<core::GlobalIndex> rbuf(static_cast<std::size_t>(rdisp.back() + rcount.back()));
    MPI_Alltoallv(sbuf.data(), scount.data(), sdisp.data(), core::mpiType<core::GlobalIndex>(), rbuf.data(), rcount.data(),
                  rdisp.data(), core::mpiType<core::GlobalIndex>(), comm); //recieve ghosst rows from other subdomains
    cand.insert(cand.end(), rbuf.begin(), rbuf.end());

    std::sort(cand.begin(), cand.end());
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
    ghosts_ = std::move(cand); //sorted, non duplicated ghosts

    //neighbor discovery: for every ghost, find the owner and the local index of the ghost in the owner subdomain
    //neighbor is alreasy sorted at the end of the loop
    std::vector<int> neighbors;
    std::vector<core::LocalIndex> rstart, rcnt;
    for (std::size_t k = 0; k < ghosts_.size(); ++k) {
        const int q = core::ownerOf(sys_.ownership, ghosts_[k]);
        if (neighbors.empty() || neighbors.back() != q) {
            neighbors.push_back(q);
            rstart.push_back(n_owned_ + static_cast<core::LocalIndex>(k));
            rcnt.push_back(0);
        }
        ++rcnt.back();
    }
    const core::LocalIndex n_local = n_owned_ + static_cast<core::LocalIndex>(ghosts_.size());
    const std::size_t nn = neighbors.size(); 

    // Temporary pattern (no send lists yet) used to exchange the request lists.
    halo_.setup(comm, neighbors, std::vector<std::vector<core::LocalIndex>>(nn), rstart, rcnt, n_local);
    std::vector<std::vector<core::GlobalIndex>> req(nn), got;
    for (std::size_t k = 0; k < nn; ++k)
        req[k].assign(ghosts_.begin() + (rstart[k] - n_owned_), ghosts_.begin() + (rstart[k] - n_owned_ + rcnt[k]));
    halo_.exchange(req, got);

    std::vector<std::vector<core::LocalIndex>> send_idx(nn);
    for (std::size_t k = 0; k < nn; ++k) {
        send_idx[k].reserve(got[k].size());
        for (core::GlobalIndex g : got[k]) {
            if (g < begin_ || g >= end)
                throw std::logic_error("Subdomain: neighbour requested an unknown this process does not own");
            send_idx[k].push_back(static_cast<core::LocalIndex>(g - begin_));
        }
    }
    halo_.setup(comm, neighbors, std::move(send_idx), rstart, rcnt, n_local);
}

void Subdomain::buildMatrices(const std::vector<std::vector<core::GlobalIndex>>& rcols,
                              const std::vector<std::vector<core::Scalar>>& rvals)
{
    const core::LocalIndex n_local = nLocal();
    const core::LocalIndex n_ghost = nGhost();
    std::vector<core::Triplet> tii, town;
    tii.reserve(sys_.col.size() * 2);
    town.reserve(sys_.col.size());

    //Owned rows: every column belongs to Omega_i by construction of the overlap.
    for (core::LocalIndex l = 0; l < n_owned_; ++l) {
        for (core::GlobalIndex k = sys_.row_ptr[static_cast<std::size_t>(l)]; k < sys_.row_ptr[static_cast<std::size_t>(l) + 1]; ++k) {
            const core::LocalIndex c = localIndex(sys_.col[static_cast<std::size_t>(k)]);
            if (c < 0) throw std::logic_error("Subdomain: owned-row column outside the overlap");
            tii.emplace_back(l, c, sys_.val[static_cast<std::size_t>(k)]);
            town.emplace_back(l, c, sys_.val[static_cast<std::size_t>(k)]);
        }
    }

    //Ghost rows: columns inside Omega_i go to A_ii (block A_Gamma, A_GammaI);
    //columns in Omega^c_i form A_Gammac and only their absolute row sums are kept.
    lump_ = core::Vec::Zero(n_ghost);
    for (int k = 0; k < halo_.numNeighbors(); ++k) {
        const auto& cols = rcols[static_cast<std::size_t>(k)];
        const auto& vals = rvals[static_cast<std::size_t>(k)];
        std::size_t pc = 0, pv = 0;
        for (core::LocalIndex j = 0; j < halo_.recvCount(k); ++j) {
            const core::LocalIndex row = halo_.recvStart(k) + j;
            const core::GlobalIndex len = cols[pc++];
            for (core::GlobalIndex q = 0; q < len; ++q, ++pc, ++pv) {
                const core::LocalIndex c = localIndex(cols[pc]);
                if (c >= 0)
                    tii.emplace_back(row, c, vals[pv]);
                else
                    lump_[row - n_owned_] += std::abs(vals[pv]);
            }
        }
    }
    Aii_.resize(n_local, n_local);
    Aii_.setFromTriplets(tii.begin(), tii.end());
    Aii_.makeCompressed();
    Aowned_.resize(n_owned_, n_local);
    Aowned_.setFromTriplets(town.begin(), town.end());
    Aowned_.makeCompressed();
}

void Subdomain::buildPartitionOfUnity(const std::string& pou)
{
    const core::LocalIndex n_local = nLocal();
    if (pou == "boolean") {
        // D_i = 1 on Omega_I_i, 0 on Omega_Gamma_i: every unknown is counted once, by its owner.
        boolean_ = true;
        D_ = core::Vec::Zero(n_local);
        D_.head(n_owned_).setOnes();
    } else if (pou == "multiplicity") {
        // D_i(j) = 1 / #{subdomains containing j}. The owner counts how many
        // neighbours hold the unknown as ghost, then the count is propagated.
        boolean_ = false;
        core::Vec mult = core::Vec::Zero(n_local);
        mult.head(n_owned_).setOnes();
        for (int k = 0; k < halo_.numNeighbors(); ++k)
            for (core::LocalIndex l : halo_.sendIndices(k)) mult[l] += 1.0;
        halo_.forward(mult);
        D_ = mult.cwiseInverse();
    } else {
        throw std::invalid_argument("unknown partition of unity '" + pou + "'");
    }
}

} // namespace schwarz2lvl::part
