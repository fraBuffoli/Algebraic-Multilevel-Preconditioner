/**
 * @file subdomain.hpp
 * @brief Overlapping subdomain Omega_i = [Omega_I_i, Omega_Gamma_i] built by each process on its own.
 */
#ifndef SCHWARZ2LVL_PARTITION_SUBDOMAIN_HPP
#define SCHWARZ2LVL_PARTITION_SUBDOMAIN_HPP

#include "halo_exchange.hpp"
#include "local_system.hpp"
#include "types.hpp"

#include <mpi.h>

#include <string>
#include <vector>

namespace schwarz2lvl::part {

/**
 * @class Subdomain
 * @brief Everything process i knows about its overlapping subdomain.
 *
 * Built collectively from the LocalSystem only (owned rows): no process ever
 * accesses global data. Construction steps:
 *  1. **Overlap.** Omega_Gamma_i is the set of nodes at distance one from
 *     Omega_I_i in the undirected graph G(A + A^T): the off-process columns
 *     of the owned rows (A part) plus the remote rows that have a column in
 *     Omega_I_i (A^T part; discovered with one MPI_Alltoall of counts and one
 *     MPI_Alltoallv). The resulting neighbour relation is symmetric.
 *  2. **Halo.** Every process tells the owners which of their unknowns it
 *     holds as ghosts; this defines the HaloExchange pattern.
 *  3. **Ghost rows.** The owners send the full rows A(Omega_Gamma_i, :) of the
 *     ghost unknowns (the only extra matrix data needed by the method).
 *  4. **Local matrices.** A_ii = R_i A R_i^T (column-major, to be factorized),
 *     A(Omega_I_i, Omega_i) (row-major, used by the distributed product), and
 *     the lumping sums s_i(j) = sum_{k in Omega^c_i} |A(j,k)| of Definition 3.1.
 *  5. **Partition of unity** D_i (sum_i R_i^T D_i R_i = I).
 *
 * Local numbering: owned unknowns [0, n_I) in increasing global order,
 * then ghosts [n_I, n_I + n_Gamma) in increasing global order.
 */
class Subdomain {
public:
    /**
     * @brief Builds the overlapping subdomain (collective).
     * @param sys Local system (owned rows) of the calling process.
     * @param pou Partition of unity: "boolean" or "multiplicity".
     */
    Subdomain(const core::LocalSystem& sys, const std::string& pou);

    /// @brief Communicator.
    MPI_Comm comm() const { return sys_.comm; }
    /// @brief The owned rows this subdomain was built from.
    const core::LocalSystem& system() const { return sys_; }
    /// @brief Number of owned unknowns n_I.
    core::LocalIndex nOwned() const { return n_owned_; }
    /// @brief Number of ghost (overlap) unknowns n_Gamma.
    core::LocalIndex nGhost() const { return static_cast<core::LocalIndex>(ghosts_.size()); }
    /// @brief Size n_i = n_I + n_Gamma of the overlapping subdomain.
    core::LocalIndex nLocal() const { return n_owned_ + nGhost(); }
    
    /// @brief Global index of local unknown @p l.
    core::GlobalIndex globalId(core::LocalIndex l) const;
    /// @brief Local index of global unknown @p g, or -1 if g is not in Omega_i.
    core::LocalIndex localIndex(core::GlobalIndex g) const;
    
    /// @brief Sorted global indices of the ghosts.
    const std::vector<core::GlobalIndex>& ghosts() const { return ghosts_; }
    /// @brief Neighbour communication pattern.
    const linalg::HaloExchange& halo() const { return halo_; }
    /// @brief A_ii = A(Omega_i, Omega_i), column-major.
    const core::SpMat& Aii() const { return Aii_; }
    /// @brief A(Omega_I_i, Omega_i): owned rows with local column indices, row-major.
    const core::SpMatRow& Aowned() const { return Aowned_; }
    /// @brief Lumping sums s_i(j), j in Omega_Gamma_i (Definition 3.1), size n_Gamma.
    const core::Vec& lumpingSums() const { return lump_; }
    /// @brief Diagonal of the partition of unity D_i, size n_i.
    const core::Vec& pou() const { return D_; }
    /// @brief True for the Boolean partition of unity (D_i = 1 on Omega_I, 0 on Omega_Gamma).
    bool booleanPou() const { return boolean_; }

private:
    const core::LocalSystem& sys_;              ///< Owned rows.
    core::LocalIndex n_owned_ = 0;              ///< n_I.
    core::GlobalIndex begin_ = 0;               ///< First owned global index.
    std::vector<core::GlobalIndex> ghosts_;     ///< Sorted ghost global indices.
    linalg::HaloExchange halo_;                 ///< Communication pattern.
    core::SpMat Aii_;                           ///< A_ii.
    core::SpMatRow Aowned_;                     ///< A(Omega_I, Omega_i).
    core::Vec lump_;                            ///< Lumping sums on the ghosts.
    core::Vec D_;                               ///< Partition of unity.
    bool boolean_ = true;                       ///< Boolean partition of unity.

    void buildOverlap();
    void buildMatrices(const std::vector<std::vector<std::int64_t>>& rcv_cols,
                       const std::vector<std::vector<core::Scalar>>& rcv_vals);
    void buildPartitionOfUnity(const std::string& pou); 
};

} // namespace schwarz2lvl::part

#endif // SCHWARZ2LVL_PARTITION_SUBDOMAIN_HPP
