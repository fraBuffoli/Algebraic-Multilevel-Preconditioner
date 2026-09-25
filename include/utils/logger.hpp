/**
 * @file logger.hpp
 * @brief Minimal rank-aware logging.
 */
#ifndef SCHWARZ2LVL_UTILS_LOGGER_HPP
#define SCHWARZ2LVL_UTILS_LOGGER_HPP

#include <mpi.h>

#include <iostream>
#include <ostream>
#include <streambuf>

namespace schwarz2lvl::utils {

/**
 * @class Log
 * @brief Rank-aware output: only rank 0 of the given communicator prints.
 */
class Log {
public:
    /// @brief Sets the global verbosity level (0 quiet, 1 summary, 2 details).
    static void setVerbosity(int v) { verbosity_() = v; }

    /// @brief Current verbosity level.
    static int verbosity() { return verbosity_(); }

    /**
     * @brief Stream printing on rank 0 of @p comm, discarding elsewhere.
     * @param comm  Communicator.
     * @param level Minimum verbosity required to print.
     */
    static std::ostream& root(MPI_Comm comm = MPI_COMM_WORLD, int level = 1)
    {
        int r = 0;
        MPI_Comm_rank(comm, &r);
        return (r == 0 && verbosity_() >= level) ? std::cout : nullStream();
    }

    /// @brief A stream that discards everything.
    static std::ostream& nullStream()
    {
        static NullBuffer buf;
        static std::ostream os(&buf);
        return os;
    }

private:
    /// @brief Stream buffer discarding every character.
    class NullBuffer : public std::streambuf {
    protected:
        int overflow(int c) override { return c; }
    };

    static int& verbosity_()
    {
        static int v = 1;
        return v;
    }
};

} // namespace schwarz2lvl::utils

#endif // SCHWARZ2LVL_UTILS_LOGGER_HPP
