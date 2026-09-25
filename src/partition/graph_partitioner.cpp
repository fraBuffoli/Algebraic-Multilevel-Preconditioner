/**
 * @file graph_partitioner.cpp
 * @brief Implementation of GraphPartitioner (METIS_PartGraphKway on G(A + A^T)).
 */
#include "graph_partitioner.hpp"

#include <metis.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace schwarz2lvl::part{

std::vector<int> GraphPartitioner::partition(const CsrMatrix& A, int nparts, const Options& opt){
    const core::GlobalIndex n = A.nrows;
    const int bs = std::max(1, opt.block_size);
    if (n % bs != 0) throw std::runtime_error("GraphPartitioner: n is not a multiple of the block size");
    if (nparts <= 1) return std::vector<int>(static_cast<std::size_t>(n), 0);

    const core::GlobalIndex nb = n / bs;
    if (nb > static_cast<core::GlobalIndex>(std::numeric_limits<idx_t>::max()))
        throw std::runtime_error("GraphPartitioner: graph too large for this METIS build (idx_t)");

    std::vector<idx_t> xadj(static_cast<std::size_t>(nb) + 1, 0);
    for (core::GlobalIndex i = 0; i < n; ++i) {
        const core::GlobalIndex I = i / bs;
        for (auto k = A.row_ptr[static_cast<std::size_t>(i)]; k < A.row_ptr[static_cast<std::size_t>(i) + 1]; ++k) {
            const core::GlobalIndex J = A.col[static_cast<std::size_t>(k)] / bs;
            if (I == J) continue;
            ++xadj[static_cast<std::size_t>(I) + 1];
            ++xadj[static_cast<std::size_t>(J) + 1];
        }
    }
    std::vector<core::GlobalIndex> start(static_cast<std::size_t>(nb) + 1, 0);
    for (core::GlobalIndex I = 0; I < nb; ++I)
        start[static_cast<std::size_t>(I) + 1] = start[static_cast<std::size_t>(I)] + xadj[static_cast<std::size_t>(I) + 1];

    std::vector<idx_t> adj(static_cast<std::size_t>(start.back()));
    {
        std::vector<core::GlobalIndex> next(start.begin(), start.end() - 1);
        for (core::GlobalIndex i = 0; i < n; ++i) {
            const core::GlobalIndex I = i / bs;
            for (auto k = A.row_ptr[static_cast<std::size_t>(i)]; k < A.row_ptr[static_cast<std::size_t>(i) + 1]; ++k) {
                const core::GlobalIndex J = A.col[static_cast<std::size_t>(k)] / bs;
                if (I == J) continue;
                adj[static_cast<std::size_t>(next[static_cast<std::size_t>(I)]++)] = static_cast<idx_t>(J);
                adj[static_cast<std::size_t>(next[static_cast<std::size_t>(J)]++)] = static_cast<idx_t>(I);
            }
        }
    }
    core::GlobalIndex w = 0;
    xadj[0] = 0;
    for (core::GlobalIndex I = 0; I < nb; ++I) {
        auto b = adj.begin() + start[static_cast<std::size_t>(I)];
        auto e = adj.begin() + start[static_cast<std::size_t>(I) + 1];
        std::sort(b, e);
        auto last = std::unique(b, e);
        for (auto it = b; it != last; ++it) adj[static_cast<std::size_t>(w++)] = *it;
        if (w > static_cast<core::GlobalIndex>(std::numeric_limits<idx_t>::max()))
            throw std::runtime_error("GraphPartitioner: too many edges for this METIS build (idx_t)");
        xadj[static_cast<std::size_t>(I) + 1] = static_cast<idx_t>(w);
    }
    adj.resize(static_cast<std::size_t>(w));
    adj.shrink_to_fit();

    std::vector<idx_t> vwgt;
    if (opt.vertex_weights) {
        vwgt.assign(static_cast<std::size_t>(nb), 0);
        for (core::GlobalIndex i = 0; i < n; ++i)
            vwgt[static_cast<std::size_t>(i / bs)] += static_cast<idx_t>(
                A.row_ptr[static_cast<std::size_t>(i) + 1] - A.row_ptr[static_cast<std::size_t>(i)]);
    }

    //METIS partitioning
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_OBJTYPE] = (opt.objective == "vol") ? METIS_OBJTYPE_VOL : METIS_OBJTYPE_CUT;
    options[METIS_OPTION_SEED] = 0; // deterministic partitions (reproducible experiments)

    idx_t nvtxs = static_cast<idx_t>(nb), ncon = 1, np = static_cast<idx_t>(nparts), objval = 0;
    std::vector<idx_t> part(static_cast<std::size_t>(nb), 0);
    const int status = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adj.data(),
                                           vwgt.empty() ? nullptr : vwgt.data(), nullptr, nullptr, &np, nullptr,
                                           nullptr, options, &objval, part.data());
    if (status != METIS_OK) throw std::runtime_error("METIS_PartGraphKway failed (code " + std::to_string(status) + ")");

    std::vector<int> out(static_cast<std::size_t>(n));
    for (core::GlobalIndex i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = static_cast<int>(part[static_cast<std::size_t>(i / bs)]);
    return out;
}

} // namespace schwarz2lvl::part