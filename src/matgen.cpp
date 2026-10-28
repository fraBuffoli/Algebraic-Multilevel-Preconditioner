/**
 * @file matgen.cpp
 * @brief Sequential tool writing the built-in test problems in Matrix Market format.
 *
 * Usage: `matgen NAME NX [NY [NZ]] [nu=V] [ncomp=C] [rho=R nb=N radius=R cfl=C t=T seed=S source=K] [out=PREFIX]`
 * writes PREFIX.mtx (matrix) and PREFIX_rhs.mtx (generated right-hand side), e.g. to compare
 * with PETSc/PCHPDDM, or with OpenFOAM (openfoam/bubblyPressureFoam -writeMatrix), on exactly
 * the same systems. rho, nb, radius, cfl, t (pseudo-time = step), seed and source (alpha | buoyancy:
 * right-hand side, see MatrixGenerator): bubbles generators.
 */
#include "matrix_generators.hpp"
#include "matrix_market_io.hpp"

#include <iostream>
#include <string>

using namespace schwarz2lvl;

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "usage: matgen NAME NX [NY [NZ]] [nu=V] [ncomp=C] [rho=R nb=N radius=R cfl=C t=T seed=S\n"
                     "              source=alpha|buoyancy]\n"
                     "              [out=PREFIX]\n"
                     "  NAME: laplace2d laplace3d convdiff2d convdiff3d coupled2d coupled3d bubbles2d bubbles3d\n";
        return 1;
    }
    try {
        const std::string name = argv[1];
        int dims[3] = {std::stoi(argv[2]), 0, 0};
        utils::GeneratorOptions o;
        std::string prefix;
        int nd = 1;
        for (int i = 3; i < argc; ++i) {
            const std::string a = argv[i];
            const std::string v = a.substr(a.find('=') + 1);
            if (a.rfind("nu=", 0) == 0) o.nu = std::stod(v);
            else if (a.rfind("ncomp=", 0) == 0) o.ncomp = std::stoi(v);
            else if (a.rfind("rho=", 0) == 0) o.rho_ratio = std::stod(v);
            else if (a.rfind("nb=", 0) == 0) o.nbubbles = std::stoi(v);
            else if (a.rfind("radius=", 0) == 0) o.radius = std::stod(v);
            else if (a.rfind("cfl=", 0) == 0) o.cfl = std::stod(v);
            else if (a.rfind("t=", 0) == 0) o.time = std::stod(v);
            else if (a.rfind("seed=", 0) == 0) o.seed = static_cast<unsigned>(std::stoul(v));
            else if (a.rfind("source=", 0) == 0) o.source = v;
            else if (a.rfind("out=", 0) == 0) prefix = v;
            else if (nd < 3) dims[nd++] = std::stoi(a);
            else throw std::invalid_argument("unexpected argument " + a);
        }
        if (dims[1] == 0) dims[1] = dims[0];
        if (dims[2] == 0) dims[2] = dims[0];
        if (prefix.empty()) prefix = name + "_" + std::to_string(dims[0]);
        o.nx = dims[0];
        o.ny = dims[1];
        o.nz = dims[2];
        const utils::GeneratedProblem p = utils::MatrixGenerator::generate(name, o);
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
