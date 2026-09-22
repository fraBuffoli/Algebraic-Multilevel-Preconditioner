#include "halo_exchange.hpp"
#include <mpi.h>
#include <map>
#include <unordered_map>
#include <numeric>
#include <stdexcept>

namespace schwarz2lvl {

void HaloExchange::setup(const std::vector<int>& partition_map,
                          const SubdomainTopology& topology) {
    int my_rank = 0, num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    const auto& interior = topology.getInteriorIndices();
    const auto& boundary = topology.getBoundaryIndices();
    num_interior_ = static_cast<Eigen::Index>(interior.size());

    // A: group MY boundary nodes by owner
    // request_ids[q]       = global ids I need FROM q
    // request_local_pos[q] = WHERE (in my local extended array) each of
    //                        those ids will live, in the SAME order
    std::map<int, std::vector<int>> request_ids;
    std::map<int, std::vector<int>> request_local_pos;
    for (size_t j = 0; j < boundary.size(); ++j) {
        const int global_id = boundary[j];
        const int owner = partition_map[global_id];
        request_ids[owner].push_back(global_id);
        request_local_pos[owner].push_back(static_cast<int>(num_interior_ + j));
    }

    // B: exchange REQUEST COUNTS with everyone (one Alltoall)
    std::vector<int> req_counts(num_ranks, 0);
    for (const auto& kv : request_ids) req_counts[kv.first] = static_cast<int>(kv.second.size());

    std::vector<int> incoming_counts(num_ranks, 0);
    MPI_Alltoall(req_counts.data(), 1, MPI_INT, incoming_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // C: exchange the actual REQUESTED global ids (one Alltoallv)
    std::vector<int> send_displs(num_ranks, 0), recv_displs(num_ranks, 0);
    for (int q = 1; q < num_ranks; ++q) {
        send_displs[q] = send_displs[q-1] + req_counts[q-1];
        recv_displs[q] = recv_displs[q-1] + incoming_counts[q-1];
    }
    const int total_send = std::accumulate(req_counts.begin(), req_counts.end(), 0);
    const int total_recv = std::accumulate(incoming_counts.begin(), incoming_counts.end(), 0);

    std::vector<int> send_buf(total_send);
    for (const auto& kv : request_ids) {
        std::copy(kv.second.begin(), kv.second.end(), send_buf.begin() + send_displs[kv.first]);
    }
    std::vector<int> recv_buf(total_recv);
    MPI_Alltoallv(send_buf.data(), req_counts.data(), send_displs.data(), MPI_INT,
                  recv_buf.data(), incoming_counts.data(), recv_displs.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // recv_buf now holds, per rank q, the global ids q is asking ME for.
    // By construction these MUST all be nodes I own (my interior set).
    std::unordered_map<int, int> global_to_local_interior;
    global_to_local_interior.reserve(interior.size() * 2);
    for (size_t j = 0; j < interior.size(); ++j) global_to_local_interior[interior[j]] = static_cast<int>(j);

    // D: assemble the final per-neighbor send/recv lists 
    neighbor_ranks_.clear();
    send_local_positions_.clear();
    recv_local_positions_.clear();

    for (int q = 0; q < num_ranks; ++q) {
        if (req_counts[q] == 0 && incoming_counts[q] == 0) continue; // not a neighbor
        neighbor_ranks_.push_back(q);

        std::vector<int> send_pos;
        send_pos.reserve(incoming_counts[q]);
        for (int k = 0; k < incoming_counts[q]; ++k) {
            const int gid = recv_buf[recv_displs[q] + k];
            const auto it = global_to_local_interior.find(gid);
            if (it == global_to_local_interior.end()) {
                throw std::runtime_error("HaloExchange Error: rank " + std::to_string(q) +
                                          " requested global id " + std::to_string(gid) +
                                          " which is not one of my interior nodes -- "
                                          "partition_map is inconsistent across ranks.");
            }
            send_pos.push_back(it->second);
        }
        send_local_positions_.push_back(std::move(send_pos));

        const auto rit = request_local_pos.find(q);
        recv_local_positions_.push_back(rit != request_local_pos.end() ? rit->second : std::vector<int>{});
    }
}

void HaloExchange::exchange(VectorType& local_values) const {
    const int num_neighbors = static_cast<int>(neighbor_ranks_.size());
    if (num_neighbors == 0) return;

    std::vector<std::vector<double>> send_buffers(num_neighbors);
    std::vector<std::vector<double>> recv_buffers(num_neighbors);
    std::vector<MPI_Request> requests;
    requests.reserve(2 * num_neighbors);

    // Post every receive BEFORE any send: avoids relying on any particular
    // arrival order and keeps this deadlock-free regardless of message
    // sizes or network timing (standard safe non-blocking pattern).
    for (int k = 0; k < num_neighbors; ++k) {
        recv_buffers[k].resize(recv_local_positions_[k].size());
        if (!recv_buffers[k].empty()) {
            MPI_Request req;
            MPI_Irecv(recv_buffers[k].data(), static_cast<int>(recv_buffers[k].size()), MPI_DOUBLE,
                      neighbor_ranks_[k], 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
    }

    for (int k = 0; k < num_neighbors; ++k) {
        send_buffers[k].resize(send_local_positions_[k].size());
        for (size_t j = 0; j < send_local_positions_[k].size(); ++j) {
            send_buffers[k][j] = local_values(send_local_positions_[k][j]);
        }
        if (!send_buffers[k].empty()) {
            MPI_Request req;
            MPI_Isend(send_buffers[k].data(), static_cast<int>(send_buffers[k].size()), MPI_DOUBLE,
                      neighbor_ranks_[k], 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
    }

    if (!requests.empty()) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }

    for (int k = 0; k < num_neighbors; ++k) {
        for (size_t j = 0; j < recv_local_positions_[k].size(); ++j) {
            local_values(recv_local_positions_[k][j]) = recv_buffers[k][j];
        }
    }
}

void HaloExchange::scatterAddToOwner(const VectorType& local_values, VectorType& owned_result) const {
    // My own direct contribution to my interior nodes -- always present, by
    // construction (a subdomain always contributes to its own interior
    // rows).
    owned_result = local_values.head(num_interior_);

    const int num_neighbors = static_cast<int>(neighbor_ranks_.size());
    if (num_neighbors == 0) return;

    // Roles reversed relative to exchange(): what there was "send"
    // (send_local_positions_, my interior positions) here is "receive AND
    // SUM"; what there was "receive" (recv_local_positions_, my boundary
    // positions) here is "pack AND SEND". Tag 1 keeps this fully distinct
    // from exchange() (tag 0), in case the two are ever in flight together.
    std::vector<std::vector<double>> send_buffers(num_neighbors);
    std::vector<std::vector<double>> recv_buffers(num_neighbors);
    std::vector<MPI_Request> requests;
    requests.reserve(2 * num_neighbors);

    for (int k = 0; k < num_neighbors; ++k) {
        recv_buffers[k].resize(send_local_positions_[k].size());
        if (!recv_buffers[k].empty()) {
            MPI_Request req;
            MPI_Irecv(recv_buffers[k].data(), static_cast<int>(recv_buffers[k].size()), MPI_DOUBLE,
                      neighbor_ranks_[k], 1, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
    }

    for (int k = 0; k < num_neighbors; ++k) {
        send_buffers[k].resize(recv_local_positions_[k].size());
        for (size_t j = 0; j < recv_local_positions_[k].size(); ++j) {
            send_buffers[k][j] = local_values(recv_local_positions_[k][j]);
        }
        if (!send_buffers[k].empty()) {
            MPI_Request req;
            MPI_Isend(send_buffers[k].data(), static_cast<int>(send_buffers[k].size()), MPI_DOUBLE,
                      neighbor_ranks_[k], 1, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
    }

    if (!requests.empty()) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }

    for (int k = 0; k < num_neighbors; ++k) {
        for (size_t j = 0; j < send_local_positions_[k].size(); ++j) {
            owned_result(send_local_positions_[k][j]) += recv_buffers[k][j];
        }
    }
}

} // namespace schwarz2lvl