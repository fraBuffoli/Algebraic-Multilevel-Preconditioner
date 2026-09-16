/**
 * @file timer.hpp
 * @brief Lightweight wall-clock timing utilities for scaling studies.
 *
 * The scalability analysis (strong/weak scaling on G100) needs a clean
 * breakdown of where time goes: partitioning, local eigensolves, coarse
 * assembly, factorizations, and the GMRES solve itself. Rather than
 * sprinkling `MPI_Wtime()` calls everywhere, every timed section is
 * registered here under a name, and `TimerRegistry::report()` prints (or
 * exports as CSV) a summary at the end of the run.
 */

#ifndef SCHWARZ2LVL_TIMER_HPP
#define SCHWARZ2LVL_TIMER_HPP

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

namespace schwarz2lvl {

// accumulator: total elapsed time and number of times the section was called 
struct TimedSection {
    double total_seconds = 0.0;
    long   call_count    = 0;
};

/**
 * @class TimerRegistry
 * @brief Global (per-process) registry of named timed sections.
 *
 * Usage:
 * @code
 * {
 *   ScopedTimer t("setup.local_eigensolve");
 *   // ... do work ...
 * } // destructor records elapsed time under that name
 * TimerRegistry::instance().report(std::cout, rank);
 * @endcode
 */
class TimerRegistry {
 public:
    static TimerRegistry& instance() {
    static TimerRegistry registry;
    return registry;
    }

    void addSample(const std::string& name, double seconds) {
    auto& section = sections_[name];
    section.total_seconds += seconds;
    section.call_count += 1;
    }

    // Human-readable report to an arbitrary stream (e.g. std::cout).
    void report(std::ostream& os, int rank) const {
    os << "== Timing report (rank " << rank << ") ==\n";
    for (const auto& [name, section] : sections_) {
        os << "  " << std::left << std::setw(32) << name << " total=" << std::right
            << std::setw(10) << std::fixed << std::setprecision(4) << section.total_seconds
            << " s   calls=" << section.call_count << "\n";
    }
    }

    // Machine-readable CSV export, one row per section, prefixed with the
    // MPI rank and the number of ranks in the run (so multiple runs at
    // different N can be concatenated into a single scaling dataset).
    void writeCsv(const std::string& path, int rank, int num_ranks) const {
    std::ofstream out(path, std::ios::app);
    for (const auto& [name, section] : sections_) {
        out << num_ranks << ',' << rank << ',' << name << ',' << section.total_seconds << ','
            << section.call_count << '\n';
    }
    }

    void reset() { sections_.clear(); }

 private:
    TimerRegistry() = default;
    std::map<std::string, TimedSection> sections_;
};

/**
 * @class ScopedTimer
 * @brief RAII stopwatch: records elapsed wall-clock time into TimerRegistry
 * @param name Name of the timed section.
 */
class ScopedTimer {
 public:
    explicit ScopedTimer(std::string name)
        : name_(std::move(name)), start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
    const auto elapsed = std::chrono::steady_clock::now() - start_;
    const double seconds = std::chrono::duration<double>(elapsed).count();
    TimerRegistry::instance().addSample(name_, seconds);
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace schwarz2lvl

#endif  // SCHWARZ2LVL_TIMER_HPP