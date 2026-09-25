/**
 * @file timer.cpp
 * @brief Implementation of TimerRegistry and ScopedTimer.
 */
#include "timer.hpp"

#include <iomanip>
#include <ostream>
#include <sstream>

namespace schwarz2lvl::utils {

TimerRegistry& TimerRegistry::instance()
{
    static TimerRegistry reg;
    return reg;
}

void TimerRegistry::add(const std::string& name, double seconds)
{
    auto it = seconds_.find(name);
    if (it == seconds_.end()) {
        order_.push_back(name);
        seconds_[name] = seconds;
    } else {
        it->second += seconds;
    }
}

double TimerRegistry::get(const std::string& name) const
{
    auto it = seconds_.find(name);
    return it == seconds_.end() ? 0.0 : it->second;
}

void TimerRegistry::clear()
{
    order_.clear();
    seconds_.clear();
}

double TimerRegistry::maxOverRanks(MPI_Comm comm, const std::string& name) const
{
    double v = get(name), out = 0.0;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_MAX, comm);
    return out;
}

void TimerRegistry::report(MPI_Comm comm, std::ostream& os) const
{
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Broadcast the phase names of rank 0 (newline separated).
    std::string names;
    if (rank == 0) {
        std::ostringstream ss;
        for (const auto& n : order_) ss << n << '\n';
        names = ss.str();
    }
    int len = static_cast<int>(names.size());
    MPI_Bcast(&len, 1, MPI_INT, 0, comm);
    names.resize(static_cast<std::size_t>(len));
    MPI_Bcast(names.data(), len, MPI_CHAR, 0, comm);

    std::vector<std::string> list;
    std::istringstream is(names);
    for (std::string line; std::getline(is, line);) list.push_back(line);

    std::vector<double> local(list.size()), vmin(list.size()), vmax(list.size()), vsum(list.size());
    for (std::size_t i = 0; i < list.size(); ++i) local[i] = get(list[i]);
    const int n = static_cast<int>(list.size());
    MPI_Reduce(local.data(), vmin.data(), n, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(local.data(), vmax.data(), n, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(local.data(), vsum.data(), n, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (rank == 0) {
        os << "Timings [s] over " << size << " ranks\n";
        os << "  " << std::left << std::setw(38) << "phase" << std::right << std::setw(12) << "min"
           << std::setw(12) << "avg" << std::setw(12) << "max" << "\n";
        for (std::size_t i = 0; i < list.size(); ++i) {
            os << "  " << std::left << std::setw(38) << list[i] << std::right << std::fixed << std::setprecision(4)
               << std::setw(12) << vmin[i] << std::setw(12) << vsum[i] / size << std::setw(12) << vmax[i] << "\n";
        }
        os << std::defaultfloat;
    }
}

ScopedTimer::ScopedTimer(std::string name) : name_(std::move(name)), start_(MPI_Wtime()) {}

ScopedTimer::~ScopedTimer() { TimerRegistry::instance().add(name_, MPI_Wtime() - start_); }

} // namespace schwarz2lvl::utils
