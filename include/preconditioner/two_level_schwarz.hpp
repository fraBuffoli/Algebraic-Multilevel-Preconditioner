/**
 * @file two_level_schwarz.hpp
 * @brief Two-level Schwarz preconditioners with the spectral coarse space of Section 3.
 */
#ifndef SCHWARZ2LVL_PRECONDITIONER_TWO_LEVEL_SCHWARZ_HPP
#define SCHWARZ2LVL_PRECONDITIONER_TWO_LEVEL_SCHWARZ_HPP

#include "coarse_space.hpp"
#include "config.hpp"
#include "distributed_matrix.hpp"
#include "local_eigensolver.hpp"
#include "one_level_schwarz.hpp"
#include "preconditioner.hpp"

#include <iosfwd>
#include <memory>

namespace schwarz2lvl::prec {

/// @brief Global statistics of the coarse space (identical on every rank).
struct CoarseStats {
    core::GlobalIndex n0 = 0;          ///< Coarse dimension.
    int m_min = 0, m_max = 0;    ///< Min / max local contribution m_i.
    double m_avg = 0.0;          ///< Average m_i.
    int kernel_subdomains = 0;   ///< Subdomains with dim ker(A~_ii) > 0.
    int kernel_dim_max = 0;      ///< Max dim ker(A~_ii).
    core::GlobalIndex kernel_part_total = 0; ///< Sum of dim (L∩K)^⊥_K.
    int dense_subdomains = 0;    ///< Subdomains solved with the dense path.
    int not_converged = 0;       ///< Subdomains where Arnoldi did not fully converge.
    long matvecs_max = 0;        ///< Max applications of the projected operator.
};

/**
 * @brief \f$ M^{-1}_{\star,additive} = R_0^T A_{00}^{-1} R_0 + M^{-1}_\star \f$ (2.2) or
 *        \f$ M^{-1}_{\star,deflated} = R_0^T A_{00}^{-1} R_0 + M^{-1}_\star (I - A R_0^T A_{00}^{-1} R_0) \f$ (2.4),
 *        with \f$ \star \f$ = RAS or ASM.
 *
 * Setup (each process on its own subdomain, then one collective coarse assembly):
 *  1. LU of A_ii (OneLevelSchwarz);
 *  2. A~_ii (Definition 3.1);
 *  3. Z_i from (3.1)-(3.2) (LocalEigensolver);
 *  4. V_i = orth(D_i Z_i), coarse numbering, A_00 = R_0 A R_0^T and its
 *     distributed factorization (CoarseSpace).
 * If the coarse space is empty (n0 = 0) the preconditioner reduces to the one-level one.
 */
class TwoLevelSchwarz final : public Preconditioner {
public:
    /// @brief Coarse correction.
    enum class Correction { Deflated, Additive };

    /**
     * @brief Builds the preconditioner (collective).
     * @param sd  Subdomain.
     * @param A   Distributed operator (used by the deflated correction).
     * @param cfg Configuration (one-level variant, tau, nev, solvers, ...).
     */
    TwoLevelSchwarz(const part::Subdomain& sd, const linalg::DistributedMatrix& A, const core::SolverConfig& cfg);

    void apply(const core::Vec& r, core::Vec& z) const override;
    std::string name() const override;

    /// @brief Coarse space statistics.
    const CoarseStats& stats() const { return stats_; }
    /// @brief Result of the local eigensolver on this rank.
    const LocalEigensolverResult& localEigen() const { return eig_; }
    /// @brief One-level part.
    const OneLevelSchwarz& oneLevel() const { return *one_; }
    /// @brief Coarse space.
    const CoarseSpace& coarse() const { return *coarse_; }
    /// @brief Prints the coarse space statistics (rank 0).
    void printStats(std::ostream& os) const;

private:
    const part::Subdomain& sd_;                     ///< Subdomain.
    const linalg::DistributedMatrix& A_;            ///< Global operator.
    Correction correction_;                         ///< Deflated or additive.
    std::unique_ptr<OneLevelSchwarz> one_;          ///< One-level preconditioner.
    std::unique_ptr<CoarseSpace> coarse_;           ///< Coarse space.
    LocalEigensolverResult eig_;                    ///< Local eigensolver output.
    CoarseStats stats_;                             ///< Statistics.
    mutable core::Vec x0_, w_, z1_;                 ///< Work vectors.
};

} // namespace schwarz2lvl::prec

#endif // SCHWARZ2LVL_PRECONDITIONER_TWO_LEVEL_SCHWARZ_HPP
