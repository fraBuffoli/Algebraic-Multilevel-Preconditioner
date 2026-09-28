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
    std::vector<std::vector<core::Scalar>> svals(static_cast<std::size_t>(nn)), rvals;
    for (int k = 0; k < nn; ++k) {
        for (core::LocalIndex l : halo_.sendIndices(k)) {
            const core::GlobalIndex b = sys_.row_ptr[static_cast<std::size_t>(l)];
            const core::GlobalIndex e = sys_.row_ptr[static_cast<std::size_t>(l) + 1];
            scols[static_cast<std::size_t>(k)].push_back(e - b);
            for (core::GlobalIndex q = b; q < e; ++q) {
                scols[static_cast<std::size_t>(k)].push_back(sys_.col[static_cast<std::size_t>(q)]);
                svals[static_cast<std::size_t>(k)].push_back(sys_.val[static_cast<std::size_t>(q)]);
            }
        }
    }
    halo_.exchange(scols, rcols);
    halo_.exchange(svals, rvals);

    buildMatrices(rcols, rvals);
    buildPartitionOfUnity(pou);
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

    std::vector<int> scount(static_cast<std::size_t>(nranks), 0), rcount(static_cast<std::size_t>(nranks), 0);
    for (const auto& pr : notify) ++scount[static_cast<std::size_t>(pr.first)]; //scount[q] = number of notifications sent to q
    MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, comm); //rcount[q] = number of notifications FROM q
    const std::vector<int> sdisp = core::displacements(scount), rdisp = core::displacements(rcount);
    std::vector<core::GlobalIndex> sbuf(notify.size());
    for (std::size_t k = 0; k < notify.size(); ++k) sbuf[k] = notify[k].second;
    std::vector<core::GlobalIndex> rbuf(static_cast<std::size_t>(rdisp.back() + rcount.back()));
    MPI_Alltoallv(sbuf.data(), scount.data(), sdisp.data(), core::mpiType<core::GlobalIndex>(), rbuf.data(), rcount.data(),
                  rdisp.data(), core::mpiType<core::GlobalIndex>(), comm);
    cand.insert(cand.end(), rbuf.begin(), rbuf.end());

    std::sort(cand.begin(), cand.end());
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
    ghosts_ = std::move(cand); //sorted, non duplicated ghosts

    // A_ji : i is a column owned by row j of another subdomain! 
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
