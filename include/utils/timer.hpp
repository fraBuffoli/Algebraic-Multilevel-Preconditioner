/**
 * @file timer.hpp
 * @brief Wall-clock timers with parallel (min/avg/max over ranks) reporting.
 */
#ifndef SCHWARZ2LVL_UTILS_TIMER_HPP
#define SCHWARZ2LVL_UTILS_TIMER_HPP

#include <mpi.h>

#include <iosfwd>
#include <map>
#include <string>
#include <vector>

namespace schwarz2lvl::utils {

/**
 * @class TimerRegistry
 * @brief Global registry accumulating the elapsed time of named phases.
 *
 * Phases are reported in order of first registration. The registry is a
 * process-wide singleton so that any class can time its own sub-phases.
 */
class TimerRegistry {
public:
    /// @brief Access to the singleton.
    static TimerRegistry& instance();

    /**
     * @brief Adds @p seconds to phase @p name.
     * @param name    Phase name (dot-separated hierarchy, e.g. "setup.eigensolve").
     * @param seconds Elapsed time.
     */
    void add(const std::string& name, double seconds);

    /// @brief Accumulated time of phase @p name on this rank (0 if unknown).
    double get(const std::string& name) const;

    /// @brief Removes every recorded phase.
    void clear();

    /**
     * @brief Prints min/avg/max over the ranks of @p comm (collective).
     *
     * The list of phases of rank 0 is broadcast, so phases executed only by
     * some ranks are reported with 0 on the others.
     * @param comm Communicator.
     * @param os   Output stream (used on rank 0 only).
     */
    void report(MPI_Comm comm, std::ostream& os) const;

    /**
     * @brief Returns the maximum over ranks of phase @p name (collective).
     * @param comm Communicator.
     * @param name Phase name.
     */
    double maxOverRanks(MPI_Comm comm, const std::string& name) const;

private:
    TimerRegistry() = default;
    std::vector<std::string> order_;          ///< Registration order.
    std::map<std::string, double> seconds_;   ///< Accumulated seconds.
};

/**
 * @class ScopedTimer
 * @brief RAII timer: measures the lifetime of the object with MPI_Wtime and
 *        stores it in the TimerRegistry.
 */
class ScopedTimer {
public:
    /// @brief Starts timing phase @p name.
    explicit ScopedTimer(std::string name);
    /// @brief Stops timing and records the elapsed time.
    ~ScopedTimer();
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string name_; ///< Phase name.
    double start_;     ///< Start time (MPI_Wtime).
};

} // namespace schwarz2lvl::utils

#endif // SCHWARZ2LVL_UTILS_TIMER_HPP
