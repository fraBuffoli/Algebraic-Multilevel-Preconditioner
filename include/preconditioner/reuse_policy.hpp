/**
 * @file reuse_policy.hpp
 * @brief When to refactorize and when to rebuild the preconditioner in a sequence of solves.
 */
#ifndef SCHWARZ2LVL_PRECONDITIONER_REUSE_POLICY_HPP
#define SCHWARZ2LVL_PRECONDITIONER_REUSE_POLICY_HPP

#include "preconditioner.hpp"

#include <string>
#include <vector>

namespace schwarz2lvl::prec {

/**
 * @class ReusePolicy
 * @brief Chooses the UpdateLevel before each solve of a sequence with a fixed sparsity pattern.
 *
 * Modes:
 *  - Off:      Rebuild before every solve (what happens when a new preconditioner object is
 *              created at every solve, e.g. in OpenFOAM without a persistent state). Reference.
 *  - Refactor: never Rebuild after the first setup (Reuse if the matrix did not change,
 *              Refactorize otherwise). Diagnostic: shows how the stale coarse basis ages.
 *  - Auto:     Reuse if the matrix did not change (correctors of one time step), Refactorize if
 *              it changed, Rebuild when rebuilding has become cheaper than not rebuilding:
 *
 * **Break-even rule (Auto).** After a Rebuild the coarse basis fits the matrix; as the matrix
 * drifts, the same tolerance needs more iterations. The convergence is measured in
 * iterations per decade of residual reduction, q = iterations / log10(||r_0|| / ||r||), which
 * depends little on the tolerance when every solve starts from 0. With q_ref the value of the first
 * solve after the last Rebuild, every solve "wastes"
 *   (iterations - q_ref * decades) * (time per iteration)
 * seconds because of the old basis. A Rebuild is requested when the time wasted since the last
 * Rebuild reaches the extra cost of a Rebuild over a Refactorize (both measured on the run).
 *
 * **Kinds of solves.** When the solves of a step differ (warm start: the first corrector starts from
 * the previous step, the next ones from the previous corrector; last corrector to an absolute
 * tolerance), q differs between them by 30-80% with a fresh basis (bubbles, 32^3), which the rule
 * above would count as aging. record() then takes a kind (the corrector index) and every kind has
 * its own q_ref, set by its first solve after the Rebuild: like is compared with like. Kind 0 only
 * (the default) is the single reference above.
 *
 * **Robust costs.** The cost of a Rebuild (Refactorize) is the *lower median* of all the Rebuild
 * (Refactorize) times measured so far, not the last one: a single stall (G100, P7 campaign: a
 * PARDISO factorization taking ~30 s on one rank while the others waited) would otherwise raise
 * the break-even point so much that the coarse basis aged for the rest of the run (73 iterations
 * per solve instead of ~22). With two measurements the lower median is the smaller one, so one
 * outlier is ignored as soon as a second Rebuild has been timed. An outlier in the *initial* setup
 * (one measurement only) is caught by the safety net on the convergence below, on by default.
 * This is the classical rent-or-buy decision: keep paying the small recurring cost until it
 * adds up to the one-time cost, then pay that. It never spends more than twice what the best
 * choice in hindsight would have spent, and it adapts by itself to the machine, the size of the
 * subdomains and the contrast of the problem, with no threshold to tune.
 *
 * Safety nets (Auto): Rebuild when q > growth * q_ref (`--rebuild-growth`, default 4: in the
 * P5, P5b and P7 campaigns the break-even rule alone let q reach at most 3.6 q_ref, while the two
 * runs spoiled by a stall reached 5.5-6 q_ref), or after `max_age` refactorizations (off by
 * default). A solve that did not converge always triggers a Rebuild.
 *
 * The decisions are local and purely arithmetic; Preconditioner::update() makes them
 * collective with agree() (times differ slightly between ranks).
 */
class ReusePolicy {
public:
    /// @brief Reuse mode.
    enum class Mode { Off, Refactor, Auto };

    /**
     * @param mode    Off, Refactor or Auto.
     * @param growth  Auto: also rebuild when q > growth * q_ref (<= 0: off).
     * @param max_age Auto: also rebuild after this many refactorizations since the last Rebuild (0: off).
     */
    ReusePolicy(Mode mode, double growth, int max_age) : mode_(mode), growth_(growth), max_age_(max_age) {}

    /// @brief Parses "off" / "refactor" / "auto".
    static Mode parseMode(const std::string& s);

    /**
     * @brief Level to request before the next solve.
     * @param matrix_changed True if the matrix changed since the last update of the preconditioner.
     */
    UpdateLevel decide(bool matrix_changed) const;

    /**
     * @brief Records the outcome of a solve.
     * @param done       Level performed before this solve.
     * @param iterations Krylov iterations.
     * @param reduction  ||r_final|| / ||r_0||.
     * @param converged  The solve reached its tolerance.
     * @param t_update   Wall time of the update performed before the solve [s].
     * @param t_solve    Wall time of the solve [s].
     * @param kind       Kind of solve (>= 0, e.g. the corrector index): q is compared with the q_ref of
     *                   the same kind.
     */
    void record(UpdateLevel done, int iterations, double reduction, bool converged, double t_update, double t_solve,
                int kind = 0);

    /// @brief Iterations per decade, or 0 if not defined (no iteration or no reduction).
    static double perDecade(int iterations, double reduction);

    /// @brief Reference q_ref of the solves of kind @p kind (0 = not yet measured).
    double reference(int kind = 0) const
    {
        return (kind >= 0 && static_cast<std::size_t>(kind) < ref_.size()) ? ref_[static_cast<std::size_t>(kind)] : 0.0;
    }
    /// @brief q of the last solve (0 = not defined).
    double last() const { return last_; }
    /// @brief Refactorizations since the last Rebuild.
    int age() const { return age_; }
    /// @brief Time wasted by the old coarse basis since the last Rebuild [s].
    double wasted() const { return wasted_; }
    /// @brief Extra cost of a Rebuild over a Refactorize, from the lower medians of the measured
    ///        times [s] (0 = unknown).
    double rebuildPremium() const;
    /// @brief Lower median of @p v (the element of rank floor((n-1)/2) in increasing order; 0 if empty).
    static double lowerMedian(std::vector<double> v);
    /// @brief Mode.
    Mode mode() const { return mode_; }

private:
    Mode mode_;
    double growth_;
    int max_age_;
    std::vector<double> ref_;  ///< Per kind: q of its first solve after the last Rebuild (0 = not yet).
    double last_ = 0.0;        ///< q of the last solve.
    double last_ref_ = 0.0;    ///< q_ref of the kind of the last solve.
    int age_ = 0;              ///< Refactorizations since the last Rebuild.
    bool converged_ = true;    ///< The last solve converged.
    double wasted_ = 0.0;      ///< Seconds lost to the old basis since the last Rebuild.
    std::vector<double> t_rebuild_;  ///< Measured Rebuild times (initial setup included).
    std::vector<double> t_refactor_; ///< Measured Refactorize times.
};

} // namespace schwarz2lvl::prec

#endif // SCHWARZ2LVL_PRECONDITIONER_REUSE_POLICY_HPP
