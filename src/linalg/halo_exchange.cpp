/**
 * @file halo_exchange.cpp
 * @brief Implementation of HaloExchange (nonblocking point-to-point messages).
 */
#include "halo_exchange.hpp"

#include <stdexcept>

namespace schwarz2lvl::linalg {

namespace {
constexpr int kTagForward = 501; ///< Tag of forward() messages.
constexpr int kTagReverse = 502; ///< Tag of reverseAdd() messages.
} // namespace

void HaloExchange::setup(MPI_Comm comm, std::vector<int> neighbors, std::vector<std::vector<core::LocalIndex>> send_idx,
                         std::vector<core::LocalIndex> recv_start, std::vector<core::LocalIndex> recv_count, core::LocalIndex n_local)
{
    if (send_idx.size() != neighbors.size() || recv_start.size() != neighbors.size() ||
        recv_count.size() != neighbors.size())
        throw std::invalid_argument("HaloExchange::setup: inconsistent sizes");
    comm_ = comm;
    neighbors_ = std::move(neighbors);
    send_idx_ = std::move(send_idx);
    recv_start_ = std::move(recv_start);
    recv_count_ = std::move(recv_count);
    n_local_ = n_local;
    const std::size_t nn = neighbors_.size();
    sbuf_.assign(nn, {});
    rbuf_.assign(nn, {});
    req_.assign(2 * nn, MPI_REQUEST_NULL);
}

void HaloExchange::forward(core::Vec& x) const
{
    const int nn = numNeighbors();
    for (int k = 0; k < nn; ++k) {
        const auto uk = static_cast<std::size_t>(k);
        // Receive directly into the contiguous ghost block of x.
        MPI_Irecv(x.data() + recv_start_[uk], recv_count_[uk], MPI_DOUBLE, neighbors_[uk], kTagForward, comm_,
                  &req_[uk]);
    }
    for (int k = 0; k < nn; ++k) {
        const auto uk = static_cast<std::size_t>(k);
        auto& b = sbuf_[uk];
        const auto& idx = send_idx_[uk];
        b.resize(idx.size());
        for (std::size_t j = 0; j < idx.size(); ++j) b[j] = x[idx[j]];
        MPI_Isend(b.data(), static_cast<int>(b.size()), MPI_DOUBLE, neighbors_[uk], kTagForward, comm_,
                  &req_[nn + uk]);
    }
    MPI_Waitall(2 * nn, req_.data(), MPI_STATUSES_IGNORE);
}

void HaloExchange::forward(core::Mat& X) const
{
    const int nn = numNeighbors();
    const Eigen::Index nc = X.cols();
    if (nc == 0) return;
    for (int k = 0; k < nn; ++k) {
        const auto uk = static_cast<std::size_t>(k);
        rbuf_[uk].resize(static_cast<std::size_t>(recv_count_[uk]) * static_cast<std::size_t>(nc));
        MPI_Irecv(rbuf_[uk].data(), static_cast<int>(rbuf_[uk].size()), MPI_DOUBLE, neighbors_[uk], kTagForward,
                  comm_, &req_[uk]);
    }
    for (int k = 0; k < nn; ++k) {
        const auto uk = static_cast<std::size_t>(k);
        auto& b = sbuf_[uk];
        const auto& idx = send_idx_[uk];
        b.resize(idx.size() * static_cast<std::size_t>(nc));
        // Row-wise packing: all columns of one row are contiguous.
        for (std::size_t j = 0; j < idx.size(); ++j)
            for (Eigen::Index c = 0; c < nc; ++c) b[j * static_cast<std::size_t>(nc) + static_cast<std::size_t>(c)] = X(idx[j], c);
        MPI_Isend(b.data(), static_cast<int>(b.size()), MPI_DOUBLE, neighbors_[uk], kTagForward, comm_,
                  &req_[nn + uk]);
    }
    MPI_Waitall(2 * nn, req_.data(), MPI_STATUSES_IGNORE);
    for (int k = 0; k < nn; ++k) {
        const auto uk = static_cast<std::size_t>(k);
        for (core::LocalIndex j = 0; j < recv_count_[uk]; ++j)
            for (Eigen::Index c = 0; c < nc; ++c)
                X(recv_start_[uk] + j, c) = rbuf_[uk][static_cast<std::size_t>(j) * static_cast<std::size_t>(nc) + static_cast<std::size_t>(c)];
    }
}

void HaloExchange::reverseAdd(core::Vec& x) const
{
    const int nn = numNeighbors();
    for (int k = 0; k < nn; ++k) {
        const auto uk = static_cast<std::size_t>(k);
        rbuf_[uk].resize(send_idx_[uk].size());
        MPI_Irecv(rbuf_[uk].data(), static_cast<int>(rbuf_[uk].size()), MPI_DOUBLE, neighbors_[uk], kTagReverse,
                  comm_, &req_[uk]);
    }
    for (int k = 0; k < nn; ++k) {
        const auto uk = static_cast<std::size_t>(k);
        // The ghost block is contiguous: send it without packing.
        MPI_Isend(x.data() + recv_start_[uk], recv_count_[uk], MPI_DOUBLE, neighbors_[uk], kTagReverse, comm_,
                  &req_[nn + uk]);
    }
    MPI_Waitall(2 * nn, req_.data(), MPI_STATUSES_IGNORE);
    for (int k = 0; k < nn; ++k) {
        const auto uk = static_cast<std::size_t>(k);
        const auto& idx = send_idx_[uk];
        for (std::size_t j = 0; j < idx.size(); ++j) x[idx[j]] += rbuf_[uk][j];
    }
}

} // namespace schwarz2lvl::linalg
