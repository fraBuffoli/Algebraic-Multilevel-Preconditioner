/**
* @file config.cpp
* @brief Implementation of SolverConfig
*/
#include "config.hpp"

#include <functional>
#include <iomanip>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace schwarz2lvl::core{

/// @brief Parses a boolean ("1/0/true/false/yes/no/on/off").
bool parseBool(const std::string& s)
{
    if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
    if (s == "0" || s == "false" || s == "no" || s == "off") return false;
    throw std::invalid_argument("invalid boolean value '" + s + "'");
}

SolverConfig SolverConfig::fromCommandLine(int argc, char** argv){
    SolverConfig c;
    using Setter = std::function<void(const std::string&)>;

    //Table key -> setter. Keeping it here makes it easy to extend.
    const std::map<std::string, Setter> table = {
        {"matrix", [&](const std::string& v) { c.matrix_file = v; }},
        {"rhs", [&](const std::string& v) { c.rhs = v; }},
        {"generate", [&](const std::string& v) { c.generate = v; }},
        {"nx", [&](const std::string& v) { c.nx = std::stoi(v); }},
        {"ny", [&](const std::string& v) { c.ny = std::stoi(v); }},
        {"nz", [&](const std::string& v) { c.nz = std::stoi(v); }},
        {"nu", [&](const std::string& v) { c.nu = std::stod(v); }},
        {"ncomp", [&](const std::string& v) { c.ncomp = std::stoi(v); }},
        {"block-size", [&](const std::string& v) { c.block_size = std::stoi(v); }},
        {"seed", [&](const std::string& v) { c.seed = static_cast<unsigned>(std::stoul(v)); }},
        {"metis-objective", [&](const std::string& v) { c.metis_objective = v; }},
        {"metis-vertex-weights", [&](const std::string& v) { c.metis_vertex_weights = parseBool(v); }},
        {"pou", [&](const std::string& v) { c.pou = v; }},
        {"one-level", [&](const std::string& v) { c.one_level = v; }},
        {"coarse", [&](const std::string& v) { c.coarse = v; }},
        {"tau", [&](const std::string& v) { c.tau = std::stod(v); }},
        {"nev", [&](const std::string& v) { c.nev = std::stoi(v); }},
        {"eig-tol", [&](const std::string& v) { c.eig_tol = std::stod(v); }},
        {"eig-maxit", [&](const std::string& v) { c.eig_maxit = std::stoi(v); }},
        {"eig-ncv", [&](const std::string& v) { c.eig_ncv = std::stoi(v); }},
        {"shift-rel", [&](const std::string& v) { c.shift_rel = std::stod(v); }},
        {"kernel-tol", [&](const std::string& v) { c.kernel_tol = std::stod(v); }},
        {"kernel-probe", [&](const std::string& v) { c.kernel_probe = std::stoi(v); }},
        {"dense-threshold", [&](const std::string& v) { c.dense_threshold = std::stoi(v); }},
        {"orth-tol", [&](const std::string& v) { c.orth_tol = std::stod(v); }},
        {"local-solver", [&](const std::string& v) { c.local_solver = v; }},
        {"coarse-solver", [&](const std::string& v) { c.coarse_solver = v; }},
        {"coarse-procs", [&](const std::string& v) { c.coarse_procs = std::stoi(v); }},
        {"restart", [&](const std::string& v) { c.restart = std::stoi(v); }},
        {"rtol", [&](const std::string& v) { c.rtol = std::stod(v); }},
        {"maxit", [&](const std::string& v) { c.maxit = std::stoi(v); }},
        {"verbose", [&](const std::string& v) { c.verbose = std::stoi(v); }},
        {"csv", [&](const std::string& v) { c.csv = v; }},
        {"history", [&](const std::string& v) { c.history = v; }},
        {"write-solution", [&](const std::string& v) { c.write_solution = v; }},
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") throw std::invalid_argument("help");
        if (arg.rfind("--", 0) != 0) throw std::invalid_argument("unexpected argument '" + arg + "'");
        arg = arg.substr(2);
        std::string key = arg, value;
        const auto eq = arg.find('=');
        if (eq != std::string::npos) {
            key = arg.substr(0, eq);
            value = arg.substr(eq + 1);
        } else {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for --" + key);
            value = argv[++i];
        }
        const auto it = table.find(key);
        if (it == table.end()) throw std::invalid_argument("unknown option --" + key);
        try {
            it->second(value);
        } catch (const std::invalid_argument&) {
            throw std::invalid_argument("invalid value '" + value + "' for --" + key);
        }
    }
    if (c.ny == 0) c.ny = c.nx;
    if (c.nz == 0) c.nz = c.nx;
    c.validate();
    return c;
}

void SolverConfig::validate() const
{
    if (matrix_file.empty() && generate.empty())
        throw std::invalid_argument("either --matrix or --generate must be given");
    if (pou != "boolean" && pou != "multiplicity") throw std::invalid_argument("--pou must be boolean|multiplicity");
    if (one_level != "ras" && one_level != "asm") throw std::invalid_argument("--one-level must be ras|asm");
    if (coarse != "deflated" && coarse != "additive" && coarse != "none")
        throw std::invalid_argument("--coarse must be deflated|additive|none");
    if (tau <= 0.0) throw std::invalid_argument("--tau must be > 0");
    if (nev < 0) throw std::invalid_argument("--nev must be >= 0");
    if (restart < 1) throw std::invalid_argument("--restart must be >= 1");
    if (metis_objective != "cut" && metis_objective != "vol")
        throw std::invalid_argument("--metis-objective must be cut|vol");
}

void SolverConfig::print(std::ostream& os) const
{
    os << "Configuration\n";
    auto line = [&os](const std::string& k, const auto& v) { os << "  " << std::left << std::setw(22) << k << v << "\n"; };
    if (!matrix_file.empty()) line("matrix", matrix_file);
    if (!generate.empty()) {
        std::ostringstream g;
        g << generate << " (nx=" << nx << ", ny=" << ny << ", nz=" << nz << ", nu=" << nu << ", ncomp=" << ncomp << ")";
        line("generator", g.str());
    }
    line("rhs", rhs);
    line("block size", block_size == 0 ? std::string("auto") : std::to_string(block_size));
    line("partition of unity", pou);
    line("one-level", one_level);
    line("coarse correction", coarse);
    line("tau", tau);
    line("nev", nev);
    line("local solver", local_solver);
    line("coarse solver", coarse_solver);
    line("coarse procs", coarse_procs == 0 ? std::string("all") : std::to_string(coarse_procs));
    line("GMRES restart", restart);
    line("rtol", rtol);
}

std::string SolverConfig::help()
{
    return R"(Usage: schwarz2lvl [options]
 Input:
  --matrix FILE            Matrix Market file (coordinate, real, general/symmetric)
  --rhs KIND               random | ones | generated | FILE (Matrix Market array)   [random]
  --generate NAME          laplace2d | laplace3d | convdiff2d | convdiff3d | coupled2d | coupled3d
  --nx N --ny N --nz N     generator grid size                                       [64]
  --nu V                   generator viscosity nu of Eq. (4.1)                        [1]
  --ncomp N                unknowns per cell of the coupled generators                [4]
  --block-size N           unknowns per cell (partition by cells)                     [auto]
  --seed N                 seed of the random right-hand side                         [12345]
 Partition:
  --metis-objective cut|vol                                                           [cut]
  --metis-vertex-weights 0|1                                                          [0]
 Preconditioner:
  --pou boolean|multiplicity   algebraic partition of unity D_i                       [boolean]
  --one-level ras|asm          Eq. (2.3) / Eq. (2.1)                                  [ras]
  --coarse deflated|additive|none  Eq. (2.4) / Eq. (2.2) / one-level only            [deflated]
  --tau T                  select |lambda| >= 1/T in (3.2)                            [0.6]
  --nev N                  max eigenpairs per subdomain                               [300]
  --eig-tol T --eig-maxit N --eig-ncv N                                               [1e-8, 1000, auto]
  --shift-rel S            relative shift used to factorize A~_ii                     [1e-13]
  --kernel-tol T           relative threshold of the numerical kernel of A~_ii        [1e-8]
  --kernel-probe N         initial block size of the kernel detection                 [8]
  --dense-threshold N      n_i <= N uses dense eigensolvers                           [400]
  --orth-tol T             rank threshold of the local coarse basis                   [1e-13]
  --local-solver auto|sparselu|umfpack|pardiso                                       [auto]
  --coarse-solver auto|mumps|root                                                    [auto]
  --coarse-procs N         processes of the distributed coarse factorization         [all]
 Krylov:
  --restart N --rtol T --maxit N                                                     [30, 1e-8, 1000]
 Output:
  --verbose 0|1|2 --csv FILE --history FILE --write-solution FILE
)";
}


} // namespace schwarz2lvl::core
