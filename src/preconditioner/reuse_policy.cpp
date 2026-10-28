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
    if (growth_ > 0.0 && last_ref_ > 0.0 && last_ > growth_ * last_ref_) return UpdateLevel::Rebuild;
    const double premium = rebuildPremium();
    if (premium > 0.0 && wasted_ >= premium) return UpdateLevel::Rebuild; // break-even reached
    return UpdateLevel::Refactorize;
}

void ReusePolicy::record(UpdateLevel done, int iterations, double reduction, bool converged, double t_update,
                         double t_solve, int kind)
{
    if (done == UpdateLevel::Rebuild) {
        age_ = 0;
        std::fill(ref_.begin(), ref_.end(), 0.0); // every kind needs a new reference
        last_ref_ = 0.0;
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
    const auto k = static_cast<std::size_t>(std::max(0, kind));
    if (k >= ref_.size()) ref_.resize(k + 1, 0.0);
    if (ref_[k] <= 0.0) { // first solve of this kind with a fresh coarse basis: the reference, nothing wasted
        ref_[k] = q;
        last_ref_ = q;
        return;
    }
    last_ref_ = ref_[k];
    // Iterations beyond what the fresh basis needed for the same reduction, times their cost.
    const double excess = iterations - ref_[k] * std::log10(1.0 / reduction);
    if (excess > 0.0) wasted_ += excess * t_solve / iterations;
}

} // namespace schwarz2lvl::prec
