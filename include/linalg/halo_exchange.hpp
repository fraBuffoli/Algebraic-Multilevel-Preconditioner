/**
 * @file halo_exchange.hpp
 * @brief Point-to-point neighbour communication between overlapping subdomains.
 *
 * Every process stores vectors on its overlapping subdomain Omega_i with the
 * local layout
 * \f[ x_{loc} = [\, x(\Omega_{I,i}) \;|\; x(\Omega_{\Gamma,i}) \,], \f]
 * i.e. the n_I owned unknowns first, followed by the n_Gamma ghost unknowns
 * (copies of unknowns owned by neighbouring processes). Ghosts are sorted by
 * global index, and since ownership is contiguous the ghosts owned by the same
 * neighbour form a *contiguous* block of the local vector.
 *
 * Two collective operations are provided, which are exactly what OpenFOAM
 * does with its `processorFvPatch` interfaces:
 *  - forward():    owners send the current owned values to the neighbours that
 *                  hold a ghost copy (ghost := owner value), i.e. x_loc = R_i x;
 *  - reverseAdd(): every ghost value is sent back to its owner and *added* to
 *                  the owned value, i.e. the owned part becomes (sum_j R_j^T x_j)
 *                  restricted to the owned rows.
 */
#ifndef SCHWARZ2LVL_LINALG_HALO_EXCHANGE_HPP
#define SCHWARZ2LVL_LINALG_HALO_EXCHANGE_HPP

#include "mpi_utils.hpp"
#include "types.hpp"

#include <mpi.h>

#include <vector>

namespace schwarz2lvl::linalg {

/**
 * @class HaloExchange
 * @brief Communication pattern between a process and its neighbours.
 *
 * For every neighbour k the object stores
 *  - the local indices of the *owned* unknowns that neighbour k holds as ghosts
 *    (send list, in the order of the neighbour's ghost numbering);
 *  - the contiguous range [recvStart(k), recvStart(k)+recvCount(k)) of the
 *    *ghost* unknowns owned by neighbour k.
 *
 * The neighbour relation is symmetric (overlap built on G(A + A^T)), so the
 * same neighbour list is used for sends and receives.
 */
class HaloExchange {
public:
    HaloExchange() = default;

    /**
     * @brief Defines the communication pattern.
     * @param comm        Communicator.
     * @param neighbors   Sorted list of neighbour ranks.
     * @param send_idx    For each neighbour, local indices (in [0, n_owned)) to send.
     * @param recv_start  For each neighbour, first local index of its ghost block.
     * @param recv_count  For each neighbour, size of its ghost block.
     * @param n_local     Local vector size n_i = n_owned + n_ghost.
     */
    void setup(MPI_Comm comm, std::vector<int> neighbors, std::vector<std::vector<core::LocalIndex>> send_idx,
               std::vector<core::LocalIndex> recv_start, std::vector<core::LocalIndex> recv_count, core::LocalIndex n_local);

    /**
     * @brief Ghost update: ghost entries of @p x receive the owners' values (collective).
     * @param x Local vector of size n_local (owned part is read, ghost part is written).
     */
    void forward(core::Vec& x) const;

    /**
     * @brief Multi-column ghost update (each column is treated as in forward(Vec&)).
     * @param X Local matrix with n_local rows.
     */
    void forward(core::Mat& X) const;

    /**
     * @brief Reverse accumulation: ghost entries are added to the owners' entries (collective).
     *
     * After the call the owned part of @p x contains the sum of all copies;
     * the ghost part is left unchanged.
     * @param x Local vector of size n_local.
     */
    void reverseAdd(core::Vec& x) const;

    /**
     * @brief Exchanges one variable-size buffer with every neighbour (collective).
     *
     * Sizes are exchanged first, then the payloads. Used for irregular data
     * (ghost matrix rows, sparse rows of the coarse basis).
     * @tparam T         Element type (must have an MPI datatype, see mpiType()).
     * @param send       One buffer per neighbour (same order as neighbors()).
     * @param recv       Output: one buffer per neighbour.
     */
    template <typename T>
    void exchange(const std::vector<std::vector<T>>& send, std::vector<std::vector<T>>& recv) const;

    /// @brief Neighbour ranks.
    const std::vector<int>& neighbors() const { return neighbors_; }
    /// @brief Number of neighbours.
    int numNeighbors() const { return static_cast<int>(neighbors_.size()); }
    /// @brief Owned local indices sent to neighbour @p k.
    const std::vector<core::LocalIndex>& sendIndices(int k) const { return send_idx_[static_cast<std::size_t>(k)]; }
    /// @brief First ghost local index received from neighbour @p k.
    core::LocalIndex recvStart(int k) const { return recv_start_[static_cast<std::size_t>(k)]; }
    /// @brief Number of ghosts received from neighbour @p k.
    core::LocalIndex recvCount(int k) const { return recv_count_[static_cast<std::size_t>(k)]; }
    /// @brief Communicator.
    MPI_Comm comm() const { return comm_; }

private:
    MPI_Comm comm_ = MPI_COMM_NULL;                             ///< Communicator.
    std::vector<int> neighbors_;                                ///< Neighbour ranks.
    std::vector<std::vector<core::LocalIndex>> send_idx_;       ///< Send lists (owned local indices).
    std::vector<core::LocalIndex> recv_start_;                  ///< Ghost block start per neighbour.
    std::vector<core::LocalIndex> recv_count_;                  ///< Ghost block size per neighbour.
    core::LocalIndex n_local_ = 0;                              ///< Local vector size.
    mutable std::vector<std::vector<core::Scalar>> sbuf_;       ///< Send buffers (reused).
    mutable std::vector<std::vector<core::Scalar>> rbuf_;       ///< Receive buffers (reused).
    mutable std::vector<MPI_Request> req_;                      ///< Request storage (reused).
};

/// @cond
template <typename T>
void HaloExchange::exchange(const std::vector<std::vector<T>>& send, std::vector<std::vector<T>>& recv) const
{
    const int nn = numNeighbors();
    const int tag_size = 701, tag_data = 702;
    std::vector<long long> ssize(static_cast<std::size_t>(nn)), rsize(static_cast<std::size_t>(nn));
    std::vector<MPI_Request> req(static_cast<std::size_t>(2 * nn));
    for (int k = 0; k < nn; ++k) {
        ssize[static_cast<std::size_t>(k)] = static_cast<long long>(send[static_cast<std::size_t>(k)].size());
        MPI_Irecv(&rsize[static_cast<std::size_t>(k)], 1, MPI_LONG_LONG, neighbors_[static_cast<std::size_t>(k)],
                  tag_size, comm_, &req[static_cast<std::size_t>(k)]);
        MPI_Isend(&ssize[static_cast<std::size_t>(k)], 1, MPI_LONG_LONG, neighbors_[static_cast<std::size_t>(k)],
                  tag_size, comm_, &req[static_cast<std::size_t>(nn + k)]);
    }
    MPI_Waitall(2 * nn, req.data(), MPI_STATUSES_IGNORE);
    recv.assign(static_cast<std::size_t>(nn), {});
    for (int k = 0; k < nn; ++k) {
        auto& r = recv[static_cast<std::size_t>(k)];
        r.resize(static_cast<std::size_t>(rsize[static_cast<std::size_t>(k)]));
        MPI_Irecv(r.data(), static_cast<int>(r.size()), mpiType<T>(), neighbors_[static_cast<std::size_t>(k)],
                  tag_data, comm_, &req[static_cast<std::size_t>(k)]);
        const auto& s = send[static_cast<std::size_t>(k)];
        MPI_Isend(s.data(), static_cast<int>(s.size()), mpiType<T>(), neighbors_[static_cast<std::size_t>(k)],
                  tag_data, comm_, &req[static_cast<std::size_t>(nn + k)]);
    }
    MPI_Waitall(2 * nn, req.data(), MPI_STATUSES_IGNORE);
}
/// @endcond

} // namespace schwarz2lvl::linalg

#endif // SCHWARZ2LVL_LINALG_HALO_EXCHANGE_HPP
