/**
 * @file reuse_policy.cpp
 * @brief Implementation of ReusePolicy.
 */
#include "reuse_policy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace schwarz2lvl::prec {

ReusePolicy::Mode ReusePolicy::parseMode(const std::string& s)
{
    if (s == "off") return Mode::Off;
    if (s == "refactor") return Mode::Refactor;
    if (s == "auto") return Mode::Auto;
    throw std::invalid_argument("unknown reuse mode '" + s + "' (off|refactor|auto)");
}

double ReusePolicy::perDecade(int iterations, double reduction)
{
    if (iterations <= 0 || !(reduction > 0.0) || !(reduction < 1.0)) return 0.0;
    return iterations / std::log10(1.0 / reduction);
}

double ReusePolicy::lowerMedian(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    const auto mid = v.begin() + static_cast<std::ptrdiff_t>((v.size() - 1) / 2);
    std::nth_element(v.begin(), mid, v.end());
    return *mid;
}

double ReusePolicy::rebuildPremium() const
{
    const double reb = lowerMedian(t_rebuild_);
    return (reb > 0.0) ? std::max(0.0, reb - lowerMedian(t_refactor_)) : 0.0;
}

UpdateLevel ReusePolicy::decide(bool matrix_changed) const
{
    if (mode_ == Mode::Off) return UpdateLevel::Rebuild;
    if (!matrix_changed) return UpdateLevel::Reuse;
    if (mode_ == Mode::Refactor) return UpdateLevel::Refactorize;
    if (!converged_) return UpdateLevel::Rebuild;
    if (max_age_ > 0 && age_ + 1 > max_age_) return UpdateLevel::Rebuild;
    if (growth_ > 0.0 && ref_ > 0.0 && last_ > growth_ * ref_) return UpdateLevel::Rebuild;
    const double premium = rebuildPremium();
    if (premium > 0.0 && wasted_ >= premium) return UpdateLevel::Rebuild; // break-even reached
    return UpdateLevel::Refactorize;
}

void ReusePolicy::record(UpdateLevel done, int iterations, double reduction, bool converged, double t_update,
                         double t_solve)
{
    if (done == UpdateLevel::Rebuild) {
        age_ = 0;
        need_ref_ = true;
        wasted_ = 0.0;
        if (t_update > 0.0) t_rebuild_.push_back(t_update);
    } else if (done == UpdateLevel::Refactorize) {
        ++age_;
        if (t_update > 0.0) t_refactor_.push_back(t_update);
    }
    converged_ = converged;
    const double q = perDecade(iterations, reduction);
    if (q <= 0.0) return;
    last_ = q;
    if (need_ref_) { // first solve with a fresh coarse basis: the reference, nothing wasted
        ref_ = q;
        need_ref_ = false;
        return;
    }
    // Iterations beyond what the fresh basis needed for the same reduction, times their cost.
    const double excess = iterations - ref_ * std::log10(1.0 / reduction);
    if (excess > 0.0) wasted_ += excess * t_solve / iterations;
}

} // namespace schwarz2lvl::prec
