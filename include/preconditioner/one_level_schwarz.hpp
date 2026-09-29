/**
 * @file one_level_schwarz.hpp
 * @brief One-level overlapping Schwarz preconditioners: ASM (2.1) and RAS (2.3).
 */
#ifndef SCHWARZ2LVL_PRECONDITIONER_ONE_LEVEL_SCHWARZ_HPP
#define SCHWARZ2LVL_PRECONDITIONER_ONE_LEVEL_SCHWARZ_HPP

#include "local_solver.hpp"
#include "preconditioner.hpp"
#include "subdomain.hpp"

#include <memory>
#include <string>

namespace schwarz2lvl::prec {

/**
 * @class OneLevelSchwarz
 * @brief \f$ M^{-1}_{ASM} = \sum_i R_i^T A_{ii}^{-1} R_i \f$ (2.1) or
 *        \f$ M^{-1}_{RAS} = \sum_i R_i^T D_i A_{ii}^{-1} R_i \f$ (2.3).
 *
 * Application on process i (one subdomain per process):
 *  1. \f$ r_i = R_i r \f$: owned entries + HaloExchange::forward() for the ghosts;
 *  2. \f$ y_i = A_{ii}^{-1} r_i \f$ with the exact LU factorization (LocalSolver);
 *  3. RAS: \f$ y_i \leftarrow D_i y_i \f$;
 *  4. \f$ z = \sum_j R_j^T y_j \f$: HaloExchange::reverseAdd() of the ghost part.
 *     With the Boolean partition of unity (D_i = 0 on the ghosts) step 4 is
 *     communication-free for RAS: z = y_i(Omega_I).
 */
class OneLevelSchwarz final : public Preconditioner {
public:
    /// @brief One-level variant.
    enum class Variant { RAS, ASM };

    /**
     * @brief Factorizes A_ii (collective: failures are reported on every rank).
     * @param sd           Overlapping subdomain.
     * @param variant      RAS or ASM.
     * @param local_solver Backend name (see makeLocalSolver()).
     * @throws std::runtime_error if some A_ii is singular.
     */
    OneLevelSchwarz(const part::Subdomain& sd, Variant variant, const std::string& local_solver);

    void apply(const core::Vec& r, core::Vec& z) const override;
    std::string name() const override { return variant_ == Variant::RAS ? "RAS" : "ASM"; }

    /// @brief Local factorization backend.
    const linalg::LocalSolver& localSolver() const { return *solver_; }
    /// @brief Variant.
    Variant variant() const { return variant_; }

private:
    const part::Subdomain& sd_;                     ///< Subdomain.
    Variant variant_;                               ///< RAS or ASM.
    std::unique_ptr<linalg::LocalSolver> solver_;   ///< LU of A_ii.
    mutable core::Vec rloc_, yloc_;                 ///< Work vectors on Omega_i.
};

} // namespace schwarz2lvl::prec

#endif // SCHWARZ2LVL_PRECONDITIONER_ONE_LEVEL_SCHWARZ_HPP
