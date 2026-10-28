# schwarz2lvl — algebraic two-level Schwarz preconditioner

C++17 / MPI implementation of the fully algebraic overlapping two-level Schwarz preconditioner of

> H. Al Daas, P. Jolivet, T. Rees, *Efficient Algebraic Two-Level Schwarz Preconditioner for Sparse
> Matrices*, SIAM J. Sci. Comput. 45(3), A1199–A1213, 2023. doi:10.1137/22M1469833

Project for the course *Advanced Programming for Scientific Computing* (Politecnico di Milano).

The preconditioner needs nothing but the matrix: every subdomain builds its coarse-space contribution
from a local generalized eigenproblem with the lumped matrix Ã_ii (Definition 3.1 of the paper), so
no mesh, near-kernel vectors or problem-specific information are required. One MPI process owns one
subdomain; the preconditioner is applied inside right-preconditioned restarted GMRES or BiCGStab.

The repository contains:

- the library and the driver `schwarz2lvl` (one-level RAS/ASM, two-level additive/deflated variants,
  setup reuse along a sequence of matrices, exact or AMG local solvers, several direct-solver backends);
- unit and integration tests (`make check`);
- the reproduction of the experiments of the paper and the performance campaigns run on CINECA
  Galileo100 (G100), up to 432 processes, including a comparison with the GAMG solver of OpenFOAM on
  the same pressure matrices. The results are summarized in [Results](#results).

## Repository layout

| Path | Content |
|---|---|
| `include/<module>/`, `src/<module>/` | the library, one class per header/source pair, namespace `schwarz2lvl::<module>` (see [Code organization](#code-organization)) |
| `src/main.cpp`, `src/matgen.cpp` | driver `schwarz2lvl` and generator of test matrices `matgen` |
| `test/test_*.cpp`, `test/test_utils.hpp` | test programs (MPI), run by `make check` |
| `test/paper/` | reproduction of Section 4 of the paper (scripts, results, [`COMPARISON.md`](test/paper/COMPARISON.md)) |
| `test/reuse/`, `test/cholesky/`, `test/p5/`, `test/p7/`, `test/gamg/` | performance campaigns on G100: SLURM scripts (`Allrun*.sh`), report scripts (`*.py`), raw outputs (`run_<jobid>/`) and a `RESULTS.md` each |
| `openfoam/bubblyPressureFoam/` | OpenFOAM-10 application that assembles the same pressure matrices, used as GAMG reference |
| `external/` | bundled header-only libraries: Eigen 3.4.0, Spectra 1.0.1, AMGCL (commit 1fbc24e) |
| `docs/NOTES.md` | implementation notes: design choices and the history of the optimizations |
| `matrices/` | destination of the downloaded SuiteSparse matrices (not versioned) |
| `Makefile`, `Doxyfile` | build, documentation |

## Requirements

| Dependency | Version tested | Status |
|---|---|---|
| C++17 compiler with an MPI wrapper (`mpicxx`) | GCC 13.3 + Open MPI 4.1.6 (Ubuntu 24.04); Intel MPI 2021 (oneAPI) on G100 | required |
| METIS | 5.1.0 | required |
| GNU Make | 4.3 | required |
| Eigen | 3.4.0 | required, **bundled** (`external/eigen3`) |
| Spectra | 1.0.1 | required, **bundled** (`external/spectra`) |
| AMGCL | commit 1fbc24e | optional, **bundled** (`external/amgcl`): AMG V-cycles as local solvers |
| SuiteSparse UMFPACK | 7.6.1 (UMFPACK 6) | optional: sequential LU backend for the local matrices |
| MUMPS (+ ScaLAPACK, BLAS/LAPACK) | 5.6.2 (ScaLAPACK 2.2.1) | optional: distributed coarse solver |
| Intel MKL (PARDISO, cluster_sparse_solver) | 2023.2.0 (G100) | optional: local backend and distributed coarse solver |
| Doxygen (+ Graphviz) | 1.9.8 (Graphviz 2.42) | optional: `make doc` |
| Python 3 (+ matplotlib) | 3.13 (matplotlib 3.11) | optional: only to regenerate the reports of the campaigns (standard library only; matplotlib for the plots) |

Without UMFPACK, MUMPS and MKL the code is complete: local matrices are factorized with the Eigen
solvers (SparseLU, SimplicialLLT/LDLT) and the coarse matrix is gathered and factorized on rank 0.
These libraries only add faster or distributed backends, selected at run time (`--local-solver`,
`--coarse-solver`; `auto` picks pardiso > umfpack > sparselu and mumps > cpardiso > root).

## Build

### Ubuntu 24.04 (or Debian-based distributions)

```bash
sudo apt install build-essential libopenmpi-dev openmpi-bin libmetis-dev \
                 libsuitesparse-dev libmumps-dev libscalapack-openmpi-dev   # last line: optional
make -j4          # bin/schwarz2lvl and bin/matgen
make check        # builds and runs every test on 1, 2 and 4 processes
```

The first line printed by `make` reports the configuration actually used, e.g.

```
schwarz2lvl: CXX=mpicxx  USE_MUMPS=1 USE_UMFPACK=1 USE_PARDISO=0 USE_CPARDISO=0 USE_AMGCL=1  EIGEN_INC=external/eigen3
```

Optional libraries are detected automatically: UMFPACK and MUMPS are enabled when their headers
are found, and disabled otherwise. If METIS or the MPI compiler is missing, `make` stops at once
with a message saying what to install.

### Build options

Every option can be given on the command line (`make USE_MUMPS=0`) or written in a file `make.inc`
next to the Makefile (not versioned, read automatically).

| Variable | Default | Meaning |
|---|---|---|
| `CXX` | `mpicxx` | MPI C++ compiler |
| `CXXFLAGS` | `-std=c++17 -O3 -march=native -DNDEBUG` | compile flags |
| `USE_UMFPACK`, `USE_MUMPS` | 1 if the header is found | SuiteSparse UMFPACK local LU, MUMPS coarse solver |
| `USE_PARDISO`, `USE_CPARDISO` | 0 | MKL PARDISO local solver, MKL cluster_sparse_solver coarse solver (needs `MKLROOT`, and `MKL_BLACS_LIB` for the MPI in use) |
| `USE_AMGCL` | 1 | AMG local solvers (`--local-approx amg`), bundled |
| `METIS_INC`, `METIS_LIB` | system paths, `-lmetis` | METIS location |
| `SUITESPARSE_INC`, `SUITESPARSE_LIB` | `/usr/include/suitesparse`, `-lumfpack -lamd -lsuitesparseconfig` | UMFPACK location |
| `MUMPS_INC`, `MUMPS_LIB` | system paths, `-ldmumps -lmumps_common -lpord -lscalapack-openmpi -llapack -lblas` | MUMPS location (the ScaLAPACK library name is the Debian/Ubuntu one) |
| `EIGEN_INC` | `external/eigen3` | Eigen headers (bundled copy by default) |
| `BUILD`, `BIN` | `build`, `bin` | directories of the objects and of the executables |

Compile flags are not tracked as dependencies: `make clean` is needed after changing a `USE_*`
switch. Other targets: `make tests` (test programs only), `make doc` (Doxygen HTML in
`docs/Doxygen/html`), `make clean`.

### CINECA Galileo100

The campaigns were compiled with the modules of the OpenFOAM-10 toolchain (Intel MPI 2021), METIS
5.1.0 and MKL 2023.2.0, PARDISO for the local matrices and cluster_sparse_solver for the coarse one
(G100 has no MUMPS module compatible with that toolchain). A `make.inc` equivalent to the build of
the job scripts (`build_s2l` in `test/gamg/gamg_functions.sh`):

```make
CXXFLAGS     = -std=c++17 -O3 -march=skylake-avx512 -DNDEBUG
METIS_INC    = <METIS prefix>/include
METIS_LIB    = -L<METIS prefix>/lib -Wl,-rpath,<METIS prefix>/lib -lmetis
USE_MUMPS    = 0
USE_UMFPACK  = 0
USE_PARDISO  = 1
USE_CPARDISO = 1
MKL_BLACS_LIB = -lmkl_blacs_intelmpi_lp64
```

```bash
module load profile/eng
module load autoload openfoam/10                # Intel MPI 2021 and the compilers of the toolchain
module load metis/5.1.0--gcc--10.2.0 intel-oneapi-mkl/2023.2.0   # `module show <name>`: install prefix
make -j8 all tests
```

Every job script of `test/` builds its own copy of the binaries inside its output directory and
records the loaded modules (`log.modules`) and a hash of the sources.

## Tests

```bash
make check                          # every test with 1, 2 and 4 processes
make check CHECK_NP="2 8"           # other process counts
mpirun -np 4 bin/test_preconditioner  # a single test, with its full output
```

`make check` prints one line per test and process count and exits with an error if any check failed;
the output of every run is kept in `bin/<test>.np<N>.log`. With Open MPI, `--oversubscribe` is added
automatically, so 4 processes also run on a 2-core laptop (`MPIEXEC` and `MPIEXEC_FLAGS` select
another launcher). The whole suite takes about 4 minutes on 2 cores.

The tests compare the distributed implementation with sequential references built from the global
matrix:

| Test | What is checked |
|---|---|
| `test_subdomain` | overlap, halo exchange, A_ii, lumping sums, partition of unity, distributed SpMV; symmetry detection of Matrix Market input |
| `test_preconditioner` | sparse (Arnoldi) vs dense eigensolver, kernel handling of (3.1); RAS/ASM × deflated/additive against a reference assembled on one process; GMRES convergence |
| `test_reuse` | refactorization vs fresh setup on time-dependent matrices, reuse policy, time loop, sequences read from files |
| `test_factorization` | Cholesky / LDLᵀ / LU chains, single precision, numerical refactorizations, every coarse solver of the build |
| `test_inexact` | AMG V-cycle as local solver: linearity, quality, two-level GMRES with the same coarse space |
| `test_stopping` | OpenFOAM normalised-residual stopping test of GMRES |
| `test_bicgstab` | BiCGStab: convergence, stopping at half/full steps, iteration budget, time loop |
| `test_warmstart` | p_rgh right-hand side, absolute tolerance, warm start, last-corrector tolerances |
| `test_bubbles` | bubble field shared with the OpenFOAM application, structure of the `bubbles3d` matrix |

## Running

```bash
./bin/schwarz2lvl --help        # all options with their defaults

# Convection-diffusion problem (4.1) of the paper, 3D, nu = 1e-2: one solve with GMRES(30)
mpirun -np 4 ./bin/schwarz2lvl --generate convdiff3d --nx 40 --nu 1e-2 --tau 0.3 --nev 60

# Same problem, one-level RAS only (no coarse space): more iterations, a gap that grows with the
# number of subdomains
mpirun -np 4 ./bin/schwarz2lvl --generate convdiff3d --nx 40 --nu 1e-2 --coarse none

# Pressure equation of a bubbly flow (density ratio 1000), 10 time steps x 3 correctors, with the
# setting used against GAMG: BiCGStab, AMG local solves, small coarse space, OpenFOAM stopping test,
# setup reuse between time steps
mpirun -np 4 ./bin/schwarz2lvl --generate bubbles3d --nx 32 --nbubbles 20 --radius 0.1 \
    --tau 0.3 --nev 8 --eig-tol 1e-2 --eig-ncv 12 --local-approx amg --krylov bicgstab \
    --stop-norm openfoam --rtol 1e-2 --steps 10 --reuse auto

# A Matrix Market file (coordinate, real/integer/pattern, general/symmetric/skew-symmetric)
mpirun -np 4 ./bin/schwarz2lvl --matrix matrices/thermal2.mtx --csv results.csv
```

Open MPI refuses to start more processes than cores unless `--oversubscribe` is added
(`mpirun --oversubscribe -np 4 ...` on a 2-core laptop).

Built-in problems (`--generate`): `laplace2d|3d`, `convdiff2d|3d` (Eq. (4.1) of the paper, upwind
finite volumes), `coupled2d|3d` (several coupled unknowns per cell), `bubbles2d|3d` (pressure of a
two-phase flow with rising bubbles). `./bin/matgen` writes the same problems as Matrix Market files.
The SuiteSparse matrices of Table 1 of the paper are downloaded into `matrices/` by
`test/paper/get_matrices.sh` (internet access needed).

Defaults reproduce Section 4 of the paper: zero initial guess, random right-hand side, τ = 0.6,
nev = 300, GMRES(30), relative tolerance 1e-8. Rank 0 prints the configuration, the subdomain sizes,
the coarse-space statistics (n0, eigensolver), the iterations with the true residual, and a
min/avg/max timing table of every phase. `--csv FILE` appends one line per run (the format used for
the scaling studies), `--history FILE` writes the residual history, `--steps-csv FILE` one line per
solve of a time loop. Exit code: 0 converged, 2 not converged, 1 invalid input.

## Results

All measurements are on G100 (48 cores per node, one subdomain per core). Each folder contains the
job scripts, the raw output of every job (`run_<jobid>/`: logs, CSV, plots) and a summary.

| Campaign | Question | Main outcome | Details |
|---|---|---|---|
| Paper reproduction | Does the implementation reproduce Section 4 of the paper? | Fig. 2, Table 1, Fig. 4 and Table 2 reproduced (Table 1: n0/N within 15% of the paper on 8 of 10 matrices, iterations within ±1 on 6; Fig. 4: iterations bounded in N); one failure (circuit5M_dc, setup > 20 min) | [`test/paper/COMPARISON.md`](test/paper/COMPARISON.md) |
| Setup reuse (P1) | What can be kept between the matrices of a time loop? | numerical refactorizations + rebuild of the coarse space at break-even: 5.7–7.4× cheaper per solve than a new preconditioner at every solve | [`test/reuse/RESULTS.md`](test/reuse/RESULTS.md) |
| Symmetric factorizations (P2) | Cholesky and single precision for SPD matrices | −10% to −18% cost per solve, same iterations | [`test/cholesky/RESULTS.md`](test/cholesky/RESULTS.md) |
| Coarse-space size (P5) | How many eigenvectors per subdomain? | a cap of 8 vectors (`--nev 8`): −52% to −80% cost per solve | [`test/p5/RESULTS.md`](test/p5/RESULTS.md) |
| Inexact local solvers (P7) | One AMG V-cycle instead of the exact factors of A_ii | same iterations after a rebuild, −26% to −56% cost per solve | [`test/p7/RESULTS.md`](test/p7/RESULTS.md) |
| Comparison with OpenFOAM GAMG | Is the preconditioner competitive with GAMG on the same matrices? | see below | [`test/gamg/RESULTS.md`](test/gamg/RESULTS.md) |

Comparison with GAMG (OpenFOAM 10, `faceAreaPair` agglomeration, DIC smoother: the fastest OpenFOAM
variant), pressure equation of a bubbly flow (150 bubbles, density ratio 1000), 192³–320³ cells,
240–432 processes, same matrices verified to 4e-15, setup and rebuilds included in the cost:

| Setting | GAMG cost / Schwarz cost |
|---|---|
| every solve from 0, ‖r‖₂/‖b‖₂ ≤ 1e-8 (240 processes) | 2.0× – 4.3× |
| every solve from 0, OpenFOAM `relTol 1e-2` (240–432 processes) | 1.8× – 2.7× per solve |
| PIMPLE-like loop: p_rgh source, warm start, `relTol 0.05`, last corrector to tolerance 1e-7 | 0.96× – 1.81× per time step |
| one-level RAS instead of two-level, same PIMPLE-like loop | 0.18× – 0.34× |

- The gain comes from the number of iterations, not from cheaper iterations: one Schwarz iteration
  costs about one GAMG V-cycle, but GAMG needs 2–3× more of them. The spectral coarse space contains
  the near-kernel modes that are nearly constant on each bubble, which the geometric agglomeration
  of GAMG cannot represent; GAMG iterations grow with the mesh (32 → 69 V-cycles from 192³ to 320³ at
  `relTol 1e-2`), the Schwarz ones much less (11 → 15 right after a rebuild).
- The two-level method scales better (parallel efficiency 240 → 432 processes 0.85–1.18 against
  0.56–0.96 for GAMG).
- In the more realistic PIMPLE-like loop the advantage shrinks: GAMG is cheaper on the first,
  loosely converged correctors, the Schwarz preconditioner on the last corrector (2.0–4.1×); setup
  and rebuilds of the coarse space are about 40% of its cost.
- The coarse space is essential: without it the one-level method is 3–6× slower than GAMG.

The reports can be regenerated from the committed outputs, without G100, with the Python scripts of
each folder (e.g. `cd test/gamg && python3 report_warm.py run_warm_np240_23097350
run_warm_np432_23097354 --ref run_scaling_np240_23040226 --ref run_scaling_np432_23040228`); the
`README.md` of each folder explains how the jobs were run.

## Code organization

```
include/<module>/*.hpp, src/<module>/*.cpp
├── core/            (core)    Scalar, LocalIndex (32-bit), GlobalIndex (64-bit); SolverConfig (command line); LocalSystem; MPI helpers
├── partition/       (part)    CsrMatrix, MatrixMarketIO, GraphPartitioner (METIS), DomainDecomposer, Subdomain
├── linalg/          (linalg)  HaloExchange, DistributedMatrix (SpMV), LocalSolver (SparseLU / UMFPACK / PARDISO,
│                              AMG approximate inverse), global dot products and norms
├── preconditioner/  (prec)    Preconditioner (+ UpdateLevel), OneLevelSchwarz, local block splitting, LocalEigensolver,
│                              CoarseSpace, CoarseSolver (MUMPS / cpardiso / root), TwoLevelSchwarz, ReusePolicy
├── solver/          (solv)    KrylovSolver (stopping tests, factory), Gmres, BiCgStab, runTimeLoop (sequence of solves)
└── utils/           (utils)   MatrixGenerator, BubbleField (header-only, shared with OpenFOAM), TimerRegistry/ScopedTimer, Log
```

Flow of a solve (`src/main.cpp`):

1. **Decomposition** (`DomainDecomposer`): rank 0 reads or generates the matrix, partitions the graph
   of A + Aᵀ with METIS, renumbers the unknowns and sends every rank its own rows (`LocalSystem`),
   as `decomposePar` does in OpenFOAM. After this step no process holds global data.
2. **Subdomain** (`Subdomain`): every rank builds its overlap Ω_i = [Ω_I | Ω_Γ] (distance one in the
   graph), the halo exchange pattern, A_ii, its rows of A, the lumping sums and the partition of unity D_i.
3. **One level** (`OneLevelSchwarz`): factorization of A_ii (or one AMG V-cycle); RAS (2.3) or ASM (2.1).
4. **Two levels** (`TwoLevelSchwarz`): Ã_ii (Definition 3.1) → `LocalEigensolver` (Z_i from (3.1)–(3.2),
   Arnoldi of Spectra on a projected operator, kernels handled explicitly) → `CoarseSpace`
   (V_i = orth(D_i Z_i), A_00 = R_0 A R_0ᵀ assembled from local products) → `CoarseSolver` (A_00).
   Deflated (2.4): z = x_0 + M⁻¹(r − A x_0), x_0 = R_0ᵀ A_00⁻¹ R_0 r; additive (2.2).
5. **Krylov** (`Gmres` or `BiCgStab`), the preconditioner applied on the right.

With `--steps N` the driver runs `runTimeLoop()`: N matrices with the same pattern, and
`Preconditioner::update()` redoes only what depends on the new values (numerical refactorizations,
or a rebuild of the coarse space decided by `ReusePolicy`).

## Documentation

- `make doc`: Doxygen documentation of every class and function (`docs/Doxygen/html/index.html`).
- [`docs/NOTES.md`](docs/NOTES.md): implementation choices (parallel layout, kernel handling of the
  local eigenproblems, coarse assembly, solver backends) and the optimization phases with their
  motivation.

## Limitations

Real matrices only; one subdomain per MPI process; the global matrix is read or generated on rank 0
during the decomposition; local indices are 32-bit and the MUMPS / MKL interfaces require n0 < 2³¹.

## License

MIT (see [`LICENSE`](LICENSE)). The bundled libraries keep their own licenses (Eigen: MPL 2.0,
Spectra: MPL 2.0, AMGCL: MIT).
