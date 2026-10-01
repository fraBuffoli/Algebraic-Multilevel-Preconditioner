/**
 * @file matgen.cpp
 * @brief Sequential tool writing the built-in test problems in Matrix Market format.
 *
 * Usage: `matgen NAME NX [NY [NZ]] [nu=V] [ncomp=C] [out=PREFIX]`
 * writes PREFIX.mtx (matrix) and PREFIX_rhs.mtx (boundary-condition right-hand side),
 * e.g. to compare with PETSc/PCHPDDM on exactly the same systems.
 */
#include "matrix_generators.hpp"
#include "matrix_market_io.hpp"

#include <iostream>
#include <string>

using namespace schwarz2lvl;

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "usage: matgen NAME NX [NY [NZ]] [nu=V] [ncomp=C] [out=PREFIX]\n"
                     "  NAME: laplace2d laplace3d convdiff2d convdiff3d coupled2d coupled3d\n";
        return 1;
    }
    try {
        const std::string name = argv[1];
        int dims[3] = {std::stoi(argv[2]), 0, 0};
        double nu = 1.0;
        int ncomp = 4;
        std::string prefix;
        int nd = 1;
        for (int i = 3; i < argc; ++i) {
            const std::string a = argv[i];
            if (a.rfind("nu=", 0) == 0) nu = std::stod(a.substr(3));
            else if (a.rfind("ncomp=", 0) == 0) ncomp = std::stoi(a.substr(6));
            else if (a.rfind("out=", 0) == 0) prefix = a.substr(4);
            else if (nd < 3) dims[nd++] = std::stoi(a);
            else throw std::invalid_argument("unexpected argument " + a);
        }
        if (dims[1] == 0) dims[1] = dims[0];
        if (dims[2] == 0) dims[2] = dims[0];
        if (prefix.empty()) prefix = name + "_" + std::to_string(dims[0]);
        const utils::GeneratedProblem p = utils::MatrixGenerator::generate(name, dims[0], dims[1], dims[2], nu, ncomp);
        part::MatrixMarketIO::writeMatrix(prefix + ".mtx", p.A);
        part::MatrixMarketIO::writeVector(prefix + "_rhs.mtx", p.rhs);
        std::cout << p.description << " -> " << prefix << ".mtx, " << prefix << "_rhs.mtx (block size "
                  << p.block_size << ")\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
