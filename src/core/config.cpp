/**
* @file config.cpp
* @brief Implementation of SolverConfig
*/
#include "config.hpp"

#include <functional>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace schwarz2lvl::core{

namespace {

/// @brief Parses a boolean ("1/0/true/false/yes/no/on/off").
bool parseBool(const std::string& s)
{
    if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
    if (s == "0" || s == "false" || s == "no" || s == "off") return false;
    throw std::invalid_argument("invalid boolean value '" + s + "'");
}

/**
 * @brief Strict numeric conversion: the whole string must be a number of type T.
 *
 * std::stoi("16x") returns 16 and std::stoi("99999999999") throws std::out_of_range: both
 * are turned into std::invalid_argument, so that a typo is rejected with the usage message.
 */
template <typename T, typename Convert> T parseNumber(const std::string& s, Convert convert)
{
    std::size_t pos = 0;
    T v{};
    try {
        v = convert(s, &pos);
    } catch (const std::out_of_range&) {
        throw std::invalid_argument("out of range");
    }
    if (pos != s.size()) throw std::invalid_argument("trailing characters");
    return v;
}

int toInt(const std::string& s)
{
    return parseNumber<int>(s, [](const std::string& x, std::size_t* p) { return std::stoi(x, p); });
}
long long toLongLong(const std::string& s)
{
    return parseNumber<long long>(s, [](const std::string& x, std::size_t* p) { return std::stoll(x, p); });
}
unsigned toUnsigned(const std::string& s)
{
    if (!s.empty() && s[0] == '-') throw std::invalid_argument("negative value");
    const unsigned long v =
        parseNumber<unsigned long>(s, [](const std::string& x, std::size_t* p) { return std::stoul(x, p); });
    if (v > std::numeric_limits<unsigned>::max()) throw std::invalid_argument("out of range");
    return static_cast<unsigned>(v);
}
double toDouble(const std::string& s)
{
    return parseNumber<double>(s, [](const std::string& x, std::size_t* p) { return std::stod(x, p); });
}

/// @brief True if @p v is one of @p allowed.
bool oneOf(const std::string& v, std::initializer_list<const char*> allowed)
{
    for (const char* a : allowed)
        if (v == a) return true;
    return false;
}

} // namespace

SolverConfig SolverConfig::fromCommandLine(int argc, char** argv){
    SolverConfig c;
    using Setter = std::function<void(const std::string&)>;

    // Table key -> setter. Keeping it here makes it easy to extend.
    const std::map<std::string, Setter> table = {
        {"matrix", [&](const std::string& v) { c.matrix_file = v; }},
        {"rhs", [&](const std::string& v) { c.rhs = v; }},
        {"generate", [&](const std::string& v) { c.generate = v; }},
        {"nx", [&](const std::string& v) { c.nx = toInt(v); }},
        {"ny", [&](const std::string& v) { c.ny = toInt(v); }},
        {"nz", [&](const std::string& v) { c.nz = toInt(v); }},
        {"nu", [&](const std::string& v) { c.nu = toDouble(v); }},
        {"ncomp", [&](const std::string& v) { c.ncomp = toInt(v); }},
        {"block-size", [&](const std::string& v) { c.block_size = toInt(v); }},
        {"seed", [&](const std::string& v) { c.seed = toUnsigned(v); }},
        {"rho-ratio", [&](const std::string& v) { c.rho_ratio = toDouble(v); }},
        {"nbubbles", [&](const std::string& v) { c.nbubbles = toInt(v); }},
        {"radius", [&](const std::string& v) { c.radius = toDouble(v); }},
        {"cfl", [&](const std::string& v) { c.cfl = toDouble(v); }},
        {"bubble-source", [&](const std::string& v) { c.bubble_source = v; }},
        {"pulse", [&](const std::string& v) { c.pulse = toDouble(v); }},
        {"pulse-period", [&](const std::string& v) { c.pulse_period = toInt(v); }},
        {"steps", [&](const std::string& v) { c.steps = toInt(v); }},
        {"correctors", [&](const std::string& v) { c.correctors = toInt(v); }},
        {"reuse", [&](const std::string& v) { c.reuse = v; }},
        {"rebuild-growth", [&](const std::string& v) { c.rebuild_growth = toDouble(v); }},
        {"rebuild-every", [&](const std::string& v) { c.rebuild_every = toInt(v); }},
        {"steps-csv", [&](const std::string& v) { c.steps_csv = v; }},
        {"warm-start", [&](const std::string& v) { c.warm_start = parseBool(v); }},
        {"final-rtol", [&](const std::string& v) { c.final_rtol = toDouble(v); }},
        {"final-atol", [&](const std::string& v) { c.final_atol = toDouble(v); }},
        {"metis-objective", [&](const std::string& v) { c.metis_objective = v; }},
        {"metis-vertex-weights", [&](const std::string& v) { c.metis_vertex_weights = parseBool(v); }},
        {"pou", [&](const std::string& v) { c.pou = v; }},
        {"one-level", [&](const std::string& v) { c.one_level = v; }},
        {"coarse", [&](const std::string& v) { c.coarse = v; }},
        {"tau", [&](const std::string& v) { c.tau = toDouble(v); }},
        {"nev", [&](const std::string& v) { c.nev = toInt(v); }},
        {"nev-start", [&](const std::string& v) { c.nev_start = toInt(v); }},
        {"eig-tol", [&](const std::string& v) { c.eig_tol = toDouble(v); }},
        {"eig-maxit", [&](const std::string& v) { c.eig_maxit = toInt(v); }},
        {"eig-ncv", [&](const std::string& v) { c.eig_ncv = toInt(v); }},
        {"shift-rel", [&](const std::string& v) { c.shift_rel = toDouble(v); }},
        {"kernel-tol", [&](const std::string& v) { c.kernel_tol = toDouble(v); }},
        {"kernel-probe", [&](const std::string& v) { c.kernel_probe = toInt(v); }},
        {"dense-threshold", [&](const std::string& v) { c.dense_threshold = toInt(v); }},
        {"orth-tol", [&](const std::string& v) { c.orth_tol = toDouble(v); }},
        {"local-solver", [&](const std::string& v) { c.local_solver = v; }},
        {"local-factorization", [&](const std::string& v) { c.local_factorization = v; }},
        {"local-precision", [&](const std::string& v) { c.local_precision = v; }},
        {"local-approx", [&](const std::string& v) { c.local_approx = v; }},
        {"amg-coarsening", [&](const std::string& v) { c.amg_coarsening = v; }},
        {"amg-sweeps", [&](const std::string& v) { c.amg_sweeps = toInt(v); }},
        {"amg-coarse-enough", [&](const std::string& v) { c.amg_coarse_enough = toInt(v); }},
        {"amg-reuse", [&](const std::string& v) { c.amg_reuse = parseBool(v); }},
        {"coarse-solver", [&](const std::string& v) { c.coarse_solver = v; }},
        {"coarse-factorization", [&](const std::string& v) { c.coarse_factorization = v; }},
        {"coarse-procs", [&](const std::string& v) { c.coarse_procs = toInt(v); }},
        {"coarse-root-below", [&](const std::string& v) { c.coarse_root_below = toLongLong(v); }},
        {"krylov", [&](const std::string& v) { c.krylov = v; }},
        {"restart", [&](const std::string& v) { c.restart = toInt(v); }},
        {"rtol", [&](const std::string& v) { c.rtol = toDouble(v); }},
        {"atol", [&](const std::string& v) { c.atol = toDouble(v); }},
        {"stop-norm", [&](const std::string& v) { c.stop_norm = v; }},
        {"maxit", [&](const std::string& v) { c.maxit = toInt(v); }},
        {"verbose", [&](const std::string& v) { c.verbose = toInt(v); }},
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
    // Everything that can be checked without MPI is checked here, before rank 0 reads or generates
    // the matrix: a typo must not cost a decomposition (or an eigensolve) before being reported.
    if (!generate.empty() && !oneOf(generate, {"laplace2d", "laplace3d", "convdiff2d", "convdiff3d", "coupled2d",
                                               "coupled3d", "bubbles2d", "bubbles3d"}))
        throw std::invalid_argument("--generate must be laplace2d|laplace3d|convdiff2d|convdiff3d|coupled2d|coupled3d|"
                                    "bubbles2d|bubbles3d");
    if (nx < 1 || ny < 1 || nz < 1) throw std::invalid_argument("--nx, --ny, --nz must be >= 1");
    if (ncomp < 1) throw std::invalid_argument("--ncomp must be >= 1");
    if (block_size < 0) throw std::invalid_argument("--block-size must be >= 0");
    if (!oneOf(local_solver, {"auto", "sparselu", "umfpack", "pardiso"}))
        throw std::invalid_argument("--local-solver must be auto|sparselu|umfpack|pardiso");
    if (!oneOf(coarse_solver, {"auto", "mumps", "cpardiso", "root"}))
        throw std::invalid_argument("--coarse-solver must be auto|mumps|cpardiso|root");
    if (coarse_procs < 0) throw std::invalid_argument("--coarse-procs must be >= 0");
    if (!(eig_tol > 0.0)) throw std::invalid_argument("--eig-tol must be > 0");
    if (eig_maxit < 1) throw std::invalid_argument("--eig-maxit must be >= 1");
    if (eig_ncv < 0) throw std::invalid_argument("--eig-ncv must be >= 0");
    if (!(shift_rel > 0.0)) throw std::invalid_argument("--shift-rel must be > 0");
    if (!(kernel_tol > 0.0)) throw std::invalid_argument("--kernel-tol must be > 0");
    if (kernel_probe < 1) throw std::invalid_argument("--kernel-probe must be >= 1");
    if (dense_threshold < 0) throw std::invalid_argument("--dense-threshold must be >= 0");
    if (orth_tol < 0.0) throw std::invalid_argument("--orth-tol must be >= 0");
    if (maxit < 1) throw std::invalid_argument("--maxit must be >= 1");
    if (pou != "boolean" && pou != "multiplicity") throw std::invalid_argument("--pou must be boolean|multiplicity");
    if (one_level != "ras" && one_level != "asm") throw std::invalid_argument("--one-level must be ras|asm");
    if (coarse != "deflated" && coarse != "additive" && coarse != "none")
        throw std::invalid_argument("--coarse must be deflated|additive|none");
    if (tau <= 0.0) throw std::invalid_argument("--tau must be > 0");
    if (nev < 0) throw std::invalid_argument("--nev must be >= 0");
    if (nev_start < 0) throw std::invalid_argument("--nev-start must be >= 0");
    if (krylov != "gmres" && krylov != "bicgstab") throw std::invalid_argument("--krylov must be gmres|bicgstab");
    if (restart < 1) throw std::invalid_argument("--restart must be >= 1");
    if (stop_norm != "l2" && stop_norm != "openfoam") throw std::invalid_argument("--stop-norm must be l2|openfoam");
    if (rtol < 0.0) throw std::invalid_argument("--rtol must be >= 0");
    if (atol < 0.0) throw std::invalid_argument("--atol must be >= 0");
    if ((atol > 0.0 || final_atol > 0.0) && stop_norm != "openfoam")
        throw std::invalid_argument("--atol / --final-atol need --stop-norm openfoam (tolerance on the normalised residual)");
    if (stop_norm == "l2" && (rtol == 0.0 || final_rtol == 0.0))
        throw std::invalid_argument("--rtol / --final-rtol 0 need --stop-norm openfoam with an --atol / --final-atol");
    if (stop_norm == "openfoam" && rtol == 0.0 && atol == 0.0)
        throw std::invalid_argument("--rtol 0 needs an --atol > 0");
    if (stop_norm == "openfoam" && final_rtol == 0.0 && (final_atol < 0.0 ? atol : final_atol) == 0.0)
        throw std::invalid_argument("--final-rtol 0 needs an --atol or --final-atol > 0");
    if (coarse_root_below < 0) throw std::invalid_argument("--coarse-root-below must be >= 0");
    if (metis_objective != "cut" && metis_objective != "vol")
        throw std::invalid_argument("--metis-objective must be cut|vol");
    if (local_factorization != "auto" && local_factorization != "lu" && local_factorization != "cholesky" &&
        local_factorization != "ldlt")
        throw std::invalid_argument("--local-factorization must be auto|lu|cholesky|ldlt");
    if (coarse_factorization != "auto" && coarse_factorization != "lu" && coarse_factorization != "cholesky" &&
        coarse_factorization != "ldlt")
        throw std::invalid_argument("--coarse-factorization must be auto|lu|cholesky|ldlt");
    if (local_precision != "double" && local_precision != "single")
        throw std::invalid_argument("--local-precision must be double|single");
    if (local_approx != "none" && local_approx != "amg") throw std::invalid_argument("--local-approx must be none|amg");
    if (amg_coarsening != "sa" && amg_coarsening != "rs") throw std::invalid_argument("--amg-coarsening must be sa|rs");
    if (amg_sweeps < 1) throw std::invalid_argument("--amg-sweeps must be >= 1");
    if (amg_coarse_enough < 1) throw std::invalid_argument("--amg-coarse-enough must be >= 1");
    if (rebuild_growth < 0.0) throw std::invalid_argument("--rebuild-growth must be >= 0");
    if (steps < 0) throw std::invalid_argument("--steps must be >= 0");
    if (correctors < 1) throw std::invalid_argument("--correctors must be >= 1");
    if (reuse != "off" && reuse != "refactor" && reuse != "auto")
        throw std::invalid_argument("--reuse must be off|refactor|auto");
    if (rebuild_every < 0) throw std::invalid_argument("--rebuild-every must be >= 0");
    if (rho_ratio <= 0.0) throw std::invalid_argument("--rho-ratio must be > 0");
    if (radius <= 0.0) throw std::invalid_argument("--radius must be > 0");
    if (nbubbles < 0) throw std::invalid_argument("--nbubbles must be >= 0");
    if (pulse_period < 1) throw std::invalid_argument("--pulse-period must be >= 1");
    if (bubble_source != "alpha" && bubble_source != "buoyancy")
        throw std::invalid_argument("--bubble-source must be alpha|buoyancy");
}

void SolverConfig::print(std::ostream& os) const
{
    os << "Configuration\n";
    auto line = [&os](const std::string& k, const auto& v) { os << "  " << std::left << std::setw(22) << k << v << "\n"; };
    if (!matrix_file.empty()) line("matrix", matrix_file);
    if (!generate.empty()) {
        std::ostringstream g;
        g << generate << " (nx=" << nx << ", ny=" << ny << ", nz=" << nz << ", nu=" << nu << ", ncomp=" << ncomp;
        if (generate.rfind("bubbles", 0) == 0)
            g << ", rho_ratio=" << rho_ratio << ", nbubbles=" << nbubbles << ", radius=" << radius << ", cfl=" << cfl
              << ", source=" << bubble_source;
        if (pulse != 0.0) g << ", pulse=" << pulse << ", period=" << pulse_period;
        g << ")";
        line("generator", g.str());
    }
    line("rhs", rhs);
    line("block size", block_size == 0 ? std::string("auto") : std::to_string(block_size));
    line("partition of unity", pou);
    line("one-level", one_level);
    line("coarse correction", coarse);
    line("tau", tau);
    line("nev", nev);
    line("nev start", nev_start == 0 ? std::string("nev (fixed)") : std::to_string(nev_start));
    if (local_approx == "none") {
        line("local solver", local_solver + ", factorization " + local_factorization +
                                 (local_precision == "single" ? ", A_ii in single precision" : ""));
    } else {
        std::ostringstream a;
        a << local_approx << " " << amg_coarsening << "+gs V(" << amg_sweeps << "," << amg_sweeps << "), coarsest <= "
          << amg_coarse_enough << (amg_reuse ? ", hierarchy reused" : ", full setup at every refactorization");
        if (local_precision == "single") a << ", single precision";
        line("local solver A_ii", a.str() + " (approximate; A~_ii: " + local_solver + " " + local_factorization + ")");
    }
    line("coarse solver", coarse_solver + ", factorization " + coarse_factorization);
    line("coarse procs", coarse_procs == 0 ? std::string("all") : std::to_string(coarse_procs));
    if (coarse_root_below > 0) line("coarse root below n0", coarse_root_below);
    line("Krylov", krylov == "gmres" ? "GMRES(" + std::to_string(restart) + ")" : std::string("BiCGStab"));
    {
        std::ostringstream t;
        t << rtol << (stop_norm == "openfoam" ? " (OpenFOAM normalised residual: sum|r| <= rtol sum|r0|)" : " (||r||_2 <= rtol ||b||_2)");
        line("rtol", t.str());
        if (atol > 0.0) {
            std::ostringstream a;
            a << atol << " (OpenFOAM tolerance: sum|r| / F <= atol)";
            line("atol", a.str());
        }
    }
    if (steps > 0) {
        line("steps x correctors", std::to_string(steps) + " x " + std::to_string(correctors));
        std::ostringstream r;
        r << reuse;
        if (reuse == "auto") {
            r << " (rebuild at break-even";
            if (rebuild_growth > 0.0) r << ", or when it/decade > " << rebuild_growth << " x reference";
            if (rebuild_every > 0) r << ", or after " << rebuild_every << " refactorizations";
            r << ")";
        }
        line("setup reuse", r.str());
        line("initial guess", warm_start ? std::string("previous solution (warm start)") : std::string("0"));
        if (final_rtol >= 0.0 || final_atol >= 0.0) {
            std::ostringstream f;
            f << "rtol " << (final_rtol >= 0.0 ? final_rtol : rtol) << ", atol " << (final_atol >= 0.0 ? final_atol : atol);
            line("last corrector", f.str());
        }
    }
}

std::string SolverConfig::help()
{
    return R"(Usage: schwarz2lvl [options]
 Input:
  --matrix FILE            Matrix Market file (coordinate, real, general/symmetric)
  --rhs KIND               random | ones | generated | FILE (Matrix Market array)   [random]
  --generate NAME          laplace2d | laplace3d | convdiff2d | convdiff3d | coupled2d | coupled3d |
                           bubbles2d | bubbles3d (pressure of a two-phase VoF flow)
  --nx N --ny N --nz N     generator grid size                                       [64]
  --nu V                   generator viscosity nu of Eq. (4.1)                        [1]
  --ncomp N                unknowns per cell of the coupled generators                [4]
  --rho-ratio R --nbubbles N --radius R --cfl C   bubbles generators              [1000, 50, 0.06, 0.5]
  --bubble-source alpha|buoyancy   bubbles: --rhs generated = vol x alpha, or the p_rgh source
                           of a VoF solver for the fluid at rest (-div(phig))         [alpha]
  --pulse A --pulse-period P   convdiff/coupled: convection x (1 + A sin(2 pi t / P))  [0, 20]
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
  --nev-start N            first k computed, increased until |lambda_k| < 1/tau (only if
                           nev >= 4 N); 0 = nev                                       [32]
  --eig-tol T --eig-maxit N --eig-ncv N                                               [1e-8, 1000, auto]
  --shift-rel S            relative shift used to factorize A~_ii                     [1e-13]
  --kernel-tol T           relative threshold of the numerical kernel of A~_ii        [1e-8]
  --kernel-probe N         initial block size of the kernel detection                 [8]
  --dense-threshold N      n_i <= N uses dense eigensolvers                           [400]
  --orth-tol T             rank threshold of the local coarse basis                   [1e-10]
  --local-solver auto|sparselu|umfpack|pardiso                                       [auto]
  --local-factorization auto|lu|cholesky|ldlt   symmetric A: auto = Cholesky, then LDL^T,
                           then LU if a factorization fails; nonsymmetric A: always LU  [auto]
  --local-precision double|single   factors of A_ii (one-level part) in float          [double]
  --local-approx none|amg  one-level part: exact factors of A_ii (none) or one AMG
                           V-cycle with Gauss-Seidel (amg, build USE_AMGCL=1);
                           A~_ii and A_00 stay exact                                  [none]
  --amg-coarsening sa|rs   smoothed aggregation | Ruge-Stuben                         [sa]
  --amg-sweeps N           Gauss-Seidel steps before/after the coarse correction      [1]
  --amg-coarse-enough N    maximum size of the coarsest level (skyline LU)            [500]
  --amg-reuse 0|1          refactorization keeps P, R (only R A P recomputed)          [0]
  --coarse-solver auto|mumps|cpardiso|root  distributed MUMPS / MKL PARDISO, or rank 0 [auto]
  --coarse-factorization auto|lu|cholesky|ldlt   A_00 with root/cpardiso (symmetric A:
                           auto = Cholesky, then LDL^T if it fails; mumps: SYM=2)      [auto]
  --coarse-procs N         processes of the distributed coarse factorization         [all]
  --coarse-root-below N    use the root coarse solver when n0 < N (0 = never)        [0]
 Krylov:
  --krylov gmres|bicgstab  gmres: restarted GMRES(restart), right preconditioning; bicgstab:
                           BiCGStab with right preconditioning (as OpenFOAM's PBiCGStab), two
                           applications of the preconditioner per step               [gmres]
  --restart N --rtol T --maxit N   (restart: GMRES only; maxit counts applications of the
                           preconditioner)                                            [30, 1e-8, 1000]
  --stop-norm l2|openfoam  l2: ||b - Ax||_2 <= rtol ||b||_2; openfoam: normalised residual of the
                           OpenFOAM solvers with relTol semantics, sum|b - Ax| <= rtol sum|b - Ax0|
                           (as OpenFOAM with tolerance 0, relTol rtol)                 [l2]
  --atol T                 openfoam only: also stop when sum|b - Ax| / F <= T (OpenFOAM
                           `tolerance`; F = normalisation factor of x0); rtol 0 = atol only [0]
 Sequence of solves (same pattern, values changing with the step):
  --steps N                steps of the time loop (0 = one solve)                     [0]
                           --generate: pseudo-time = step; --matrix with a printf pattern
                           (e.g. p_%04d.mtx): one file per step, starting from 0
  --correctors N           solves per step with the same matrix, new random rhs       [3]
  --reuse off|refactor|auto  off: full setup at every solve; refactor: never rebuild
                           the coarse basis; auto: reuse / refactorize, rebuild when the
                           time lost to the old basis reaches the cost of a rebuild  [auto]
  --rebuild-growth G       auto: also rebuild when it/decade > G x reference (0 = off) [4]
  --rebuild-every N        auto: also rebuild after N refactorizations (0 = off)       [0]
  --steps-csv FILE         one line per solve (step, level, times, iterations)
  --warm-start 0|1         start every solve from the previous solution (OpenFOAM) [0]
  --final-rtol T --final-atol T   tolerances of the last corrector of every step (as
                           p_rghFinal, e.g. --final-rtol 0 --final-atol 1e-7)  [rtol, atol]
 Output:
  --verbose 0|1|2 --csv FILE --history FILE --write-solution FILE
)";
}


} // namespace schwarz2lvl::core
