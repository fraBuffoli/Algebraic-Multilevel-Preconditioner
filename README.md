# schwarz2lvl — Algebraic Two-Level Schwarz Preconditioner

C++17 / MPI implementation of the fully algebraic overlapping two-level Schwarz preconditioner of

> H. Al Daas, P. Jolivet, T. Rees, *Efficient Algebraic Two-Level Schwarz Preconditioner for Sparse Matrices*,
> SIAM J. Sci. Comput. 45(3), A1199–A1213, 2023. doi:10.1137/22M1469833

The second level is a spectral coarse space built from the matrix alone: every subdomain solves a local
generalized eigenproblem involving the lumped matrix Ã_ii (Definition 3.1), which is assembled in O(n_i)
operations from the distance-one overlap. One MPI process owns one subdomain; the preconditioner is used
inside right-preconditioned restarted GMRES or BiCGStab. Real arithmetic only.

## Prerequisites

| Dependency | Status | Notes |
|---|---|---|
| C++17 compiler + MPI (`mpicxx`) | required | tested with GCC 13 + Open MPI 4.1 |
| METIS 5 | required | `-lmetis` |
| Eigen 3.4.0 | required, bundled | `external/eigen3` |
| Spectra 1.0.1 | required, bundled | `external/spectra` (header-only) |
| MUMPS (+ ScaLAPACK, BLAS/LAPACK) | optional, `USE_MUMPS=1` (default) | distributed coarse solver |
| SuiteSparse UMFPACK | optional, `USE_UMFPACK=1` (default) | local LU backend |
| Intel MKL PARDISO | optional, `USE_PARDISO=0` (default) | local LU backend |
| Intel MKL cluster_sparse_solver | optional, `USE_CPARDISO=0` (default) | distributed coarse solver (needs the MKL BLACS of your MPI, `MKL_BLACS_LIB`) |
| AMGCL (commit 1fbc24e, MIT) | optional, `USE_AMGCL=1` (default), bundled | `external/amgcl` (header-only, no Boost): AMG V-cycles as approximate local solvers (`--local-approx amg`) |
| Doxygen | optional | `make doc` |

Ubuntu 24.04 (everything, including the optional MUMPS/UMFPACK backends):

```bash
sudo apt install build-essential libopenmpi-dev openmpi-bin libmetis-dev \
                 libsuitesparse-dev libmumps-dev libscalapack-openmpi-dev
```

## Build

```bash
make -j EIGEN_INC=external/eigen3                              # bin/schwarz2lvl, bin/matgen
make -j EIGEN_INC=external/eigen3 USE_MUMPS=0 USE_UMFPACK=0    # minimal build: MPI + METIS only
make tests EIGEN_INC=external/eigen3                           # bin/test_subdomain, bin/test_preconditioner
make doc                                                       # Doxygen HTML in docs/Doxygen/html
make clean
```

- The Makefile defaults to a system Eigen (`/usr/include/eigen3`); `EIGEN_INC=external/eigen3` uses the bundled copy.
- Any variable (`CXX`, `CXXFLAGS`, `EIGEN_INC`, `METIS_INC/LIB`, `MUMPS_INC/LIB`, `SUITESPARSE_INC/LIB`, `MKL_INC/LIB`,
  `MKL_BLACS_LIB`, `BUILD`, `BIN`, `USE_*`) can be overridden on the command line or stored in a `make.inc` file next to the Makefile.
- Compile flags are not tracked as dependencies: run `make clean` after changing a `USE_*` switch.

### CINECA Galileo100

Load compiler, MPI, METIS (and optionally MUMPS / MKL) modules, then point the Makefile to them with a `make.inc`
(module names and install prefixes: `module avail`, `module show <name>`):

```make
EIGEN_INC   = external/eigen3
METIS_INC   = $(METIS_HOME)/include
METIS_LIB   = -L$(METIS_HOME)/lib -lmetis
MUMPS_INC   = $(MUMPS_HOME)/include
MUMPS_LIB   = -L$(MUMPS_HOME)/lib -ldmumps -lmumps_common -lpord <scalapack + blas/lapack>
USE_UMFPACK = 0
USE_PARDISO = 1          # MKLROOT is set by the MKL module
USE_CPARDISO = 1         # distributed coarse solver without MUMPS (MKL_BLACS_LIB = -lmkl_blacs_intelmpi_lp64)
```

G100 has no MUMPS module compatible with the OpenFOAM 10 toolchain: there the distributed coarse solver is
`cpardiso` (see `test/paper/Allrun_coarse.sh` for a complete build line).

Compute nodes have no internet access: copy the `.mtx` files beforehand. Minimal job script:

```bash
#!/bin/bash
#SBATCH --partition=g100_usr_prod
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --time=00:30:00
module load <same modules used to compile>
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # one sequential local solver per rank
mpirun ./bin/schwarz2lvl --matrix matrices/garon1.mtx --csv results.csv
```

## Usage

```bash
mpirun -np 4  ./bin/schwarz2lvl --matrix matrices/sherman4.mtx
mpirun -np 16 ./bin/schwarz2lvl --generate convdiff3d --nx 40 --nu 1e-2 --tau 0.3 --nev 60
mpirun -np 16 ./bin/schwarz2lvl --generate coupled2d --nx 200 --ncomp 4 --coarse none   # one-level RAS only
mpirun -np 16 ./bin/schwarz2lvl --generate bubbles3d --nx 48 --tau 0.3 --nev 60 --steps 20 --reuse auto  # time loop
./bin/schwarz2lvl --help                                                                 # full option list
```

Options are given as `--key value` or `--key=value`. Defaults reproduce the setup of Section 4 of the paper
(zero initial guess, random right-hand side, τ = 0.6, nev = 300, GMRES(30), rtol = 1e-8).

| Group | Options |
|---|---|
| Input | `--matrix FILE` (Matrix Market coordinate, real/integer/pattern, general/symmetric/skew) **or** `--generate laplace2d\|laplace3d\|convdiff2d\|convdiff3d\|coupled2d\|coupled3d\|bubbles2d\|bubbles3d` with `--nx --ny --nz --nu --ncomp`, `--rho-ratio --nbubbles --radius --cfl` (bubbles), `--pulse --pulse-period` (time-dependent convection); `--rhs random\|ones\|generated\|FILE`; `--block-size N` (unknowns per cell) |
| Partition | `--metis-objective cut\|vol`, `--metis-vertex-weights 0\|1` |
| Preconditioner | `--one-level ras\|asm`, `--coarse deflated\|additive\|none`, `--pou boolean\|multiplicity`, `--tau`, `--nev`, `--nev-start` |
| Local eigensolver | `--eig-tol --eig-maxit --eig-ncv`, `--shift-rel`, `--kernel-tol`, `--kernel-probe`, `--dense-threshold`, `--orth-tol` |
| Direct solvers | `--local-solver auto\|sparselu\|umfpack\|pardiso`, `--local-factorization auto\|lu\|cholesky\|ldlt`, `--local-precision double\|single`, `--coarse-solver auto\|mumps\|cpardiso\|root`, `--coarse-factorization auto\|lu\|cholesky\|ldlt`, `--coarse-procs N`, `--coarse-root-below N` |
| Approximate local solvers | `--local-approx none\|amg` (A_ii of the one-level part only), `--amg-coarsening sa\|rs`, `--amg-sweeps N`, `--amg-coarse-enough N`, `--amg-reuse 0\|1` |
| Krylov | `--krylov gmres\|bicgstab`, `--restart` (GMRES), `--rtol`, `--maxit` (applications of the preconditioner), `--stop-norm l2\|openfoam` (stopping test: ‖r‖₂ ≤ rtol ‖b‖₂, or the normalised residual of OpenFOAM, Σ\|r\| ≤ rtol Σ\|r₀\|) |
| Sequence | `--steps N` (time loop; 0 = single solve), `--correctors N`, `--reuse off\|refactor\|auto`, `--rebuild-growth G`, `--rebuild-every N`, `--steps-csv FILE`; `--matrix 'p_%04d.mtx'` reads one file per step |
| Output | `--verbose 0\|1\|2`, `--csv FILE`, `--history FILE`, `--write-solution FILE` |

`auto` picks the best backend compiled in (local: pardiso > umfpack > sparselu; coarse: mumps > cpardiso > root).

**Output.** Rank 0 prints the configuration, min/avg/max subdomain sizes (n_I, n_Γ, neighbours, nnz(A_ii)), coarse
space statistics (n0, m_i, kernel dimensions, Arnoldi convergence), Krylov iterations (applications of the
preconditioner) with the residual of the iteration and the true relative residual, and a min/avg/max timing
breakdown of every phase (the preconditioner application is split into
`solve.preconditioner.coarse.{restrict,solve,prolong}`, `.spmv` and `.local.{halo,lu}`; the Krylov part into
`solve.spmv`, `solve.orthogonalization` (GMRES) or `solve.vector_ops` (BiCGStab) and `solve.stopping_test`; the coarse solve contains
collectives, so its max also includes waiting for the slowest rank). `--csv` appends one line per run (problem, n, nnz, np,
options, n0, iterations, residual, setup / eigensolver / coarse assembly / coarse factorization / solve times),
which is the format intended for strong/weak scaling studies. `--history` writes the residual at every iteration,
`--write-solution` the solution in the original ordering. `--verbose 2` also prints, on stderr, one line per rank
as soon as its local eigensolver ends (time, n_i, n_Γ, kernel, selected vectors, operator applications), to find
the subdomains that slow down or block the setup. Exit code: 0 converged, 2 not converged, 1 invalid input.

**matgen** writes the built-in problems in Matrix Market format (e.g. to compare with PETSc/PCHPDDM, or
with OpenFOAM: `openfoam/bubblyPressureFoam -writeMatrix`):

```bash
./bin/matgen convdiff3d 40 nu=0.01 out=cd3d_40      # -> cd3d_40.mtx, cd3d_40_rhs.mtx
./bin/matgen bubbles3d 64 nb=150 radius=0.04 rho=1000 cfl=0.5 t=7 out=bub64   # bubbles at step 7
```

**Tests** compare the distributed implementation with sequential references built from the global matrix;
run them with several process counts:

```bash
mpirun -np 4 ./bin/test_subdomain        # overlap, halo, A_ii, lumping sums, partition of unity, SpMV
mpirun -np 4 ./bin/test_preconditioner   # sparse vs dense eigensolver, RAS/ASM x deflated/additive, GMRES
mpirun -np 4 ./bin/test_reuse            # setup reuse: refactorizations vs fresh setups, policy, time loop
mpirun -np 4 ./bin/test_factorization    # Cholesky / LDLT / LU chains, single precision, coarse solvers
mpirun -np 4 ./bin/test_inexact          # AMG as approximate local solver: linearity, quality, two-level GMRES
mpirun -np 1 ./bin/test_bubbles          # bubble field shared with OpenFOAM, bubbles3d matrix structure
mpirun -np 4 ./bin/test_stopping         # OpenFOAM stopping test of GMRES: exact stop, normalisation factor
mpirun -np 4 ./bin/test_bicgstab         # BiCGStab: convergence, OpenFOAM stop at half/full steps, budget, time loop
```

`matrices/` contains small SuiteSparse test matrices (sherman4, nos5, e05r0400, fidapm05/10, garon1, graham1,
lns_3937, cage7/10, windtunnel_evap2d).

## Code structure

```
include/<module>/*.hpp, src/<module>/*.cpp      one class per header/source pair, namespace schwarz2lvl::<ns>
├── core/            (core)    Scalar, LocalIndex (32-bit), GlobalIndex (64-bit); SolverConfig (CLI); LocalSystem; MPI helpers
├── partition/       (part)    CsrMatrix, MatrixMarketIO, GraphPartitioner (METIS), DomainDecomposer, Subdomain
├── linalg/          (linalg)  HaloExchange, DistributedMatrix (SpMV), LocalSolver (SparseLU/UMFPACK/PARDISO; IC/AMG
│                              approximate inverses in inexact_solver), global dot/norm
├── preconditioner/  (prec)    Preconditioner (+ UpdateLevel), OneLevelSchwarz, local block splitting, LocalEigensolver,
│                              CoarseSpace, CoarseSolver (MUMPS/cpardiso/root), TwoLevelSchwarz, ReusePolicy
├── solver/          (solv)    KrylovSolver (stopping norms, factory), Gmres, BiCgStab, runTimeLoop (sequence of solves with setup reuse)
└── utils/           (utils)   MatrixGenerator, BubbleField (header-only, shared with OpenFOAM), TimerRegistry/ScopedTimer, Log
src/main.cpp, src/matgen.cpp   drivers;   test/   MPI test programs;   external/   Eigen, Spectra, AMGCL
openfoam/bubblyPressureFoam    OpenFOAM-10 application: the bubbles3d matrices solved by the OpenFOAM solvers (GAMG reference)
```

Flow of `main.cpp`:

1. **Decomposition** — `DomainDecomposer::decompose`: rank 0 reads or generates the system, partitions G(A + Aᵀ)
   with METIS, renumbers the unknowns and sends each rank its owned rows → `LocalSystem`.
2. **Subdomain** — overlap Ω_i = [Ω_I | Ω_Γ], halo pattern, A_ii, A(Ω_I, Ω_i), lumping sums, partition of unity D_i.
3. **One level** — `OneLevelSchwarz`: exact factorization of A_ii (or an approximate inverse, `--local-approx`); RAS (2.3) or ASM (2.1).
4. **Two level** — `TwoLevelSchwarz`: Ã_ii (Def. 3.1) → `LocalEigensolver` gives Z_i (3.1)–(3.2) → `CoarseSpace`
   builds V_i = orth(D_i Z_i) and A_00 = R_0 A R_0ᵀ → `CoarseSolver` factorizes A_00.
   Deflated (2.4): z = x_0 + M₁⁻¹(r − A x_0), x_0 = R_0ᵀ A_00⁻¹ R_0 r; additive (2.2): z = R_0ᵀ A_00⁻¹ R_0 r + M₁⁻¹ r.
5. **Solve** — `Gmres` (default) or `BiCgStab` (`--krylov bicgstab`), the preconditioner applied on the right.

With `--steps N > 0` the driver runs `runTimeLoop()` instead (see [Setup reuse](#setup-reuse-between-solves-time-loops)).

## Implementation choices

**Parallel layout**
- The decomposition emulates OpenFOAM `decomposePar`: rank 0 does all global work once, then no process holds
  global data. `LocalSystem` (owned rows in CSR with global columns, contiguous ownership ranges) is the only
  interface to the preconditioner, so it can be filled directly from a processor-local `lduMatrix` + `globalIndex`.
- METIS k-way on the undirected graph G(A + Aᵀ) with a fixed seed (reproducible runs). With `block_size > 1` the graph
  is compressed by cells, so all unknowns of a cell stay in the same subdomain (coupled solvers).
- The overlap (distance one in G(A + Aᵀ)) is discovered in parallel with one `MPI_Alltoall` + `MPI_Alltoallv`;
  owners then send the full ghost rows A(Ω_Γ, :), the only extra matrix data the method needs.
- Local numbering: owned unknowns first, then ghosts sorted by global index. With contiguous ownership the ghosts of
  each neighbour form a contiguous block, received directly into the vector without unpacking.
- `HaloExchange::forward` realizes R_i, `reverseAdd` realizes Σ_j R_jᵀ (non-blocking point-to-point, buffers reused).
- Boolean partition of unity by default (D_i = 1 on Ω_I, 0 on Ω_Γ): RAS and the coarse restriction/prolongation
  become communication-free. The multiplicity PoU (1 / number of subdomains sharing an unknown) is also available.
- Errors detected on one rank are reduced/broadcast so that every rank throws instead of hanging in a collective.

**Coarse space**
- Lumping sums s_i(j) = Σ_{k∈Ω^c_i} |A(j,k)| are accumulated while the ghost rows are received; Ã_ii only changes
  the diagonal of the overlap block (missing diagonal entries are inserted).
- Ã_ii is singular for floating subdomains (zero row sums after lumping), so it is never inverted. It is factorized
  with a tiny shift σ = shift_rel·‖Ã_ii‖; the right kernel K_i (and the left kernel for nonsymmetric A) is detected
  explicitly by block inverse iteration + SVD.
- The infinite eigenvalues of (3.1), "≈ 1/ε" in the paper, are handled exactly: (L_i ∩ K_i)^⊥ in K_i is computed from
  an SVD of B_i Q and always included in Z_i.
- The finite eigenvalues come from the implicitly restarted Arnoldi method of Spectra (`GenEigsSolver`, largest
  magnitude) applied matrix-free to T = Π_K (Ã_ii + σI)⁻¹ Π_R B_i Π_K, B_i = D_i A_ii D_i: one sparse solve + one
  SpMV per iteration. Subdomains with n_i ≤ `dense_threshold` use a dense SVD/eigendecomposition path, which is
  also the reference in the tests. Selection: the nev largest, kept if |λ| ≥ 1/τ (Section 4). Since only the
  eigenvalues above 1/τ are kept, when nev ≥ 4·`--nev-start` Arnoldi computes k = `--nev-start` (32) of them
  first and increases k (at least doubling it, or up to where a power law fitted on the computed moduli reaches
  1/τ) until the smallest computed |λ| is below 1/τ, with nev as the cap: the selected vectors are the same as
  computing nev at once, but the dense Krylov work (~ n_i k² per restart, ~ k³ for the projected matrix) follows
  the number of selected vectors instead of nev (`--nev-start 0` computes nev directly, as in the paper). If the
  QR iteration on the projected Hessenberg matrix fails (strongly non-normal T, e.g. convection-dominated
  problems), Arnoldi is restarted up to twice with another starting vector and a larger ncv ("Arnoldi retried"
  in the log).
- Real arithmetic: Eigen has no complex generalized eigensolver and the target application (CFD) only produces
  real matrices. A complex-conjugate pair contributes Re(u) and Im(u), which span the same real invariant subspace.
- V_i = orth(D_i Z_i) via column-pivoted QR with a rank threshold: same span as D_i Z_i, but orthonormal columns
  give a better conditioned A_00 and drop dependent vectors. Each rank owns the m_i coarse unknowns it generates.
- A_00 is assembled without gathering anything: each rank computes W(Ω_I,:)ᵀ A(Ω_I,Ω_i) W(Ω_i,:) with local products
  and passes the triplets to the coarse solver, which sums duplicates.

**Solvers**
- Coarse: MUMPS with distributed assembled input (ICNTL(18)=3), LDLᵀ if A is symmetric, automatic retry with more
  workspace; `--coarse-procs` restricts MUMPS to group leaders to cut communication at large process counts.
  `cpardiso`: MKL cluster_sparse_solver on K = `--coarse-procs` leaders (the first rank of every group of N/K
  consecutive ranks). Each coefficient is sent to the leader owning its row, so every leader passes a contiguous
  block of rows (distributed CSR,
  iparm[39] = 2, right-hand side and solution distributed the same way: only a Gatherv/Scatterv inside each group per
  solve). Cholesky (mtype 2, checked by a probe solve) or LDLᵀ (mtype −2) if A is symmetric (see
  `--coarse-factorization` below), LU with scaling and matching otherwise, METIS ordering on the first
  process (the MPI ordering of PARDISO, iparm[1] = 10, hung in tests). With Open MPI the symmetric types crash inside
  MKL on communicators other than MPI_COMM_WORLD, so there symmetric A_00 are factorized as general matrices.
  `root` (gather on rank 0 + sequential LU) is the non-scalable fallback/reference, but for small n0 it is faster:
  the solve phase of a distributed solver has a fixed cost (collectives) that does not shrink with n0.
  `--coarse-root-below N` switches to `root` when n0 < N (the choice is collective, n0 is known on every rank);
  the CSV reports the coarse solver actually used.
- Local: backends selected at run time behind one `LocalSolver` interface. UMFPACK runs with METIS ordering and no
  iterative refinement (faster factorizations and solves on 3D problems); PARDISO is meant to run with one thread
  per rank (`MKL_NUM_THREADS=1`). PARDISO (local and `cpardiso`) also runs without iterative refinement (iparm[7] = 0;
  Eigen sets 2, which performs two refinement steps in every solve, 3-3.7x slower solves on subdomain-sized
  3D matrices): PARDISO still refines automatically when pivots were perturbed, and the log reports the maximum
  number of perturbed pivots of A_ii and Ã_ii + σI. The kernel detection uses multi-RHS solves (one per step).
  Transposed solves are needed for the left kernel of Ã_ii.
- GMRES: right preconditioning, so the stopping test is on the unpreconditioned residual as in the paper;
  classical Gram–Schmidt applied twice (CGS2), one `MPI_Allreduce` per pass instead of j+1; the true residual is
  recomputed at every restart, and convergence is declared only if it is within 10% of the tolerance (the
  GMRES residual and the true one can drift apart, e.g. with single-precision local factors).
- BiCGStab: the algorithm of OpenFOAM's PBiCGStab with fused reductions (see [BiCGStab](#bicgstab)).
- Test generators: cell-centred finite volumes for Eq. (4.1) with upwind convection, the velocity fields of
  Section 4.2 integrated exactly on the faces (zero row sums, hence singular Ã_ii that exercise the kernel handling),
  heterogeneous diffusivity, and coupled variants with unknowns interleaved per cell as in OpenFOAM coupled solvers.
- The random right-hand side is generated in the original ordering, so it does not depend on the number of processes.

## Setup reuse between solves (time loops)

In a CFD time loop the pressure matrix keeps its sparsity pattern; its values change a little at
every time step and not at all between the correctors of a step. Rebuilding the preconditioner at
every solve (what happens when a new solver object is created at every solve) throws away work that
depends only on the pattern, or only on the coarse basis. The setup is split by what each part
depends on, and `Preconditioner::update(UpdateLevel)` redoes only what is needed:

| depends on | what | Reuse | Refactorize | Rebuild |
|---|---|:--:|:--:|:--:|
| graph of A | overlap, halo, patterns of A_ii and A(Ω_I, Ω_i), partition of unity | kept | kept | kept |
| pattern of A_ii | symbolic analysis of the LU (ordering, elimination tree) | kept | kept | redone |
| values of A | A_ii, A(Ω_I, Ω_i), lumping sums (`Subdomain::updateValues`) | kept | refilled | refilled |
| values of A | numerical LU of A_ii | kept | redone | redone |
| coarse basis V | W = R_0ᵀ on Ω_i (halo exchanges), pattern and distribution of A_00 | kept | kept | redone |
| values of A and V | A_00 = R_0 A R_0ᵀ and its numerical LU | kept | redone | redone |
| values of A | Ã_ii, local eigenproblems, V | kept | **kept (old)** | redone |

- **Refactorize** keeps the coarse basis of the last Rebuild but recomputes A_00 with the current A,
  so the coarse correction stays consistent with the current matrix (`test_reuse` checks
  R_0 (r - A R_0ᵀ A_00⁻¹ R_0 r) = 0): the preconditioner is a correct preconditioner of the new
  matrix, only its coarse space is tuned to an older one. Its cost is a numerical LU of A_ii, two
  local products and a numerical LU of A_00 (no eigensolver, no halo exchange of sparse rows).
- **Numerical refactorizations** reuse the symbolic analysis: `LocalSolver::refactorize()`
  (PARDISO phase 22, UMFPACK numeric with the old symbolic object, `SparseLU::factorize`) and
  `CoarseSolver::update()` (root: same gather pattern; MUMPS: JOB = 2; cpardiso: values along the
  Alltoallv pattern of the setup, phase 22). For nonsymmetric matrices PARDISO computes scaling and
  matching in the analysis from the values of that time; if the numerical phase perturbs more pivots
  than the full one, both fall back to a full factorization.
- **Policy** (`ReusePolicy`, `--reuse`): `off` rebuilds before every solve (reference), `refactor`
  never rebuilds (diagnostic: how the coarse basis ages), `auto` reuses when the matrix did not change,
  refactorizes when it did, and rebuilds at **break-even**: each solve wastes
  (iterations − q_ref · decades) × (time per iteration) because of the old basis, q_ref being the
  iterations per decade of residual reduction right after the last rebuild; when the time wasted
  since the last rebuild reaches the measured extra cost of a rebuild over a refactorization, the
  basis is rebuilt (the rent-or-buy rule, no threshold to tune; the "at most twice the best choice"
  bound of the textbook problem assumes a constant rent, which aging is not). The costs are the
  **lower medians** of all the rebuild / refactorization times measured in the run, so a single
  stall (P7 campaign: one PARDISO factorization taking ~30 s on one rank) does not freeze the
  rebuilds. Safety nets: `--rebuild-growth G` (rebuild when q > G q_ref, default 4: never reached
  by the break-even rule in the P5/P7 campaigns, it catches an outlier in the initial setup) and
  `--rebuild-every N` (off); a solve that does not converge always triggers a rebuild. Decisions
  are made collective (`agree()`: maximum over the ranks).
- **Driver** (`runTimeLoop`): steps × correctors; the matrix values of a step come from the
  generator at pseudo-time = step or from one Matrix Market file per step (`--matrix 'p_%04d.mtx'`,
  e.g. matrices written by OpenFOAM), sent by rank 0 without repartitioning
  (`DomainDecomposer::updateValues`, pattern checked by hash); new random rhs per corrector; zero
  initial guess. Output: one line per solve (level, update and solve time, iterations, iterations
  per decade, n0, time wasted by the old basis), a summary, the timers `update.*` (refactorizations)
  and `setup.*` (initial setup and rebuilds), and `--steps-csv`.
- **Test problem** `bubbles2d/3d`: pressure equation of an incompressible two-phase flow,
  −∇·(β∇p) = f with β = 1/ρ, rising gas bubbles with a smoothed (VoF-like) interface, density ratio
  `--rho-ratio`; SPD and diagonally dominant (Lemma 3.2 of the paper applies). The pattern is fixed,
  the values change near the moving interfaces.
- `test/reuse/Allrun_reuse.sh` (G100, 5 nodes, < 1 h) compares `off`, `refactor` and `auto` on
  bubbles at two densities and two subdomain sizes, one-level RAS, and a nonsymmetric sequence;
  `plot_reuse.py` writes the report and the plots.

In an OpenFOAM port the solver object is created at every `solve()`: the persistent state
(subdomain structure, preconditioner, policy) must live outside it, e.g. in a `MeshObject` (as GAMG
stores its agglomeration) or in a registry keyed by the field name; `p_rgh` and `p_rghFinal` share it.

## Symmetric factorizations and single precision (P2)

Every application of the preconditioner solves with the factors of A_ii (one-level part) and of
A_00 (coarse part); every refactorization recomputes them. For a symmetric matrix a symmetric
factorization stores one triangular factor instead of two, so it can be done with less work and
memory. What is factorized, and how:

| matrix | symmetric A (e.g. pressure) | factorization with `auto` | precision |
|---|---|---|---|
| A_ii (one-level solves, every iteration) | SPD if A is SPD | Cholesky → LDLᵀ → LU | `--local-precision` |
| Ã_ii + σI (eigensolver operator) | PSD + shift | Cholesky → LDLᵀ → LU | always double |
| A_00 = Vᵀ A V (coarse solves, every iteration) | SPD if A is SPD | root: Cholesky → LDLᵀ → LU; cpardiso: mtype 2 → −2; mumps: SYM = 2 (request ignored) | always double |

UMFPACK has neither symmetric factorizations nor single precision: it always uses LU in double (and
says so in its name). PARDISO and the Eigen built-in solvers (SimplicialLLT / SimplicialLDLT) have both.

- **Chains.** `--local-factorization` (A_ii, Ã_ii) and `--coarse-factorization` (A_00) accept
  `auto` (= `cholesky`), `ldlt`, `lu`. Each candidate of the chain is tried until one factorizes
  successfully, and the solver stays there (also in the numerical refactorizations of a time loop:
  if a refactorization fails, e.g. the matrix is no longer positive definite, the next candidate
  is used). A symmetric factorization only reads one triangle, so it would be silently wrong for a
  nonsymmetric matrix: if A is not known to be symmetric every request becomes LU. The log reports
  the factorization actually used (`A_ii factorization`, `A~_ii factorization`, `coarse solver`) and
  on how many ranks the chain had to move on.
- **What it gains** (measured with PARDISO, one thread, on a 32³ subdomain of the bubbles problem):
  Cholesky halves the (re)factorization time; a solve still reads the factor twice (L, then Lᵀ),
  as many bytes as L and U, so it is only ~20% faster (smaller factor). LDLᵀ with Bunch–Kaufman
  pivoting is not faster than LU in sequential PARDISO: it is only a fallback. With
  cluster_sparse_solver Cholesky refactorizations were ~25-30% faster than LDLᵀ (1-2 processes).
- **Distributed Cholesky.** With more than one process MKL (2023.2) does not report a Cholesky
  factorization of a matrix that is not positive definite (no error code; NaN or wrong factors):
  every mtype 2 factorization is checked with a probe solve A_00 x = 1 (backward error < 1e-8),
  and a failure moves to mtype −2. This path cannot run with Open MPI (see above); it was tested
  with MPICH (same ABI as Intel MPI) and `test_factorization` checks it on G100.
- **Single precision** (`--local-precision single`): the factors of A_ii are stored in float (PARDISO
  iparm[27] = 1, Eigen solvers on float matrices); the right-hand side is rounded to float, solved,
  and converted back. Solves read half the bytes (~45% faster than Cholesky in double), but the
  local solves are only accurate to ~1e-7 κ(A_ii), so GMRES may need more iterations (+15-20% on
  bubbles at ρ = 1000): it pays off only when the local solves dominate the iteration. Ã_ii (the
  eigenvectors must be accurate) and A_00 stay in double. GMRES works in double precision and the
  true-residual check guarantees the requested tolerance.
- `test/cholesky/Allrun_p2.sh` (G100, 5 nodes, < 1 h) compares the P1 configuration (LU) with Cholesky,
  Cholesky + single precision, and root vs distributed A_00, on the bubbles problems of P1 and on
  SPD SuiteSparse matrices; `report_p2.py` writes the report (cost per solve and cost of every
  operation).

## Inexact local solvers (P7)

Every GMRES iteration applies A_ii⁻¹ once per subdomain. With exact factors this is the largest
cost of an iteration on large subdomains (P5, bubbles at 240 processes: 60-70% at ~30k cells per
process), and a refactorization at every time step costs a numerical Cholesky factorization. The
Cholesky factor of a 3D subdomain has 20-35 times the nonzeros of A_ii (PARDISO with METIS
ordering on a 7-point Laplacian of 30k / 69k unknowns: nnz(L) = 4.3M / 15.8M; fill-in grows like
n^(4/3)), and every solve reads it twice. `--local-approx amg` replaces A_ii⁻¹ **in the one-level part only**
with one AMG V-cycle B_i ≈ A_ii⁻¹:

M⁻¹ = Σ_i R_iᵀ D_i **B_i** R_i  (instead of Σ_i R_iᵀ D_i A_ii⁻¹ R_i),

while the coarse space (eigenproblems with Ã_ii, factorized exactly) and A_00 do not change.
Schwarz theory allows it: if B_i is spectrally equivalent to A_ii⁻¹ with constants independent of
the subdomain size, the method converges asymptotically like the one with exact local solves, with
a condition number larger by a factor of the order of those constants (Toselli-Widlund, Ch. 2;
Dolean-Jolivet-Nataf, end of Sect. 6.3). A V-cycle is spectrally equivalent for Laplace-like
operators; on the bubbles problem (ρ ratio 1000) the first solve after every rebuild needed the
same iterations as with exact factors.

| option | meaning | default |
|---|---|---|
| `--local-approx none\|amg` | exact factors of A_ii, or one AMG V-cycle (AMGCL, builtin backend, x = 0 at every application) | `none` |
| `--amg-coarsening sa\|rs` | smoothed aggregation (cheap setup, operator complexity ~1.5) or Ruge-Stuben (iterations closest to the exact solves, operator complexity ~2.7, heavier setup) | `sa` |
| `--amg-sweeps N` | Gauss-Seidel steps before (forward) and after (backward) the coarse correction: the cycle is symmetric | 1 |
| `--amg-coarse-enough N` | the coarsest level has at most N unknowns and is solved with a skyline LU | 500 |
| `--amg-reuse 0\|1` | refactorization: full AMG setup (0) or keep P and R and recompute only R A P and the smoothers (1) | 0 |
| `--local-precision single` | the whole hierarchy in float (GMRES stays in double) | double |

- **Linear and fixed.** Every application starts from zero and performs the same operations, so
  B_i is a fixed linear operator and plain GMRES remains valid (no flexible GMRES needed);
  `test_inexact` checks linearity and determinism.
- **First G100 campaign** (`test/p7/run_p7_23018578`, 240 processes, `--nev 8 --eig-tol 1e-4`):
  with `sa` + Gauss-Seidel the cost per solve was -47% at ~30k cells per process and -54% at ~70k
  (-33% / -39% against exact factors in single precision); the local solves were 3.4x / 4.3x
  cheaper, the refactorizations of A_ii 5x / 10x. At 960 processes the coarse solve dominates and the gain
  is small. What was learnt there is now in the defaults:
  - incomplete Cholesky (not spectrally equivalent: 2.4x the iterations at 7k cells per subdomain,
    8.7x at 70k), the spai0 and Chebyshev smoothers (equal or worse than Gauss-Seidel), the ILU(0)
    smoother and the options for several V-cycles and for the strength threshold (only tested
    locally, or not at all) were removed;
  - `--amg-coarse-enough 500` instead of AMGCL's 3000: a coarsest level of 1000-3000 unknowns has a
    nearly dense skyline profile, and with 17k cells per subdomain (2 levels) the AMG setup cost
    more than the exact Cholesky factorization (0.55 s against 0.23 s) and every V-cycle paid the
    dense solve;
  - `--amg-reuse 0`: with the reused hierarchy the iterations grew faster between two rebuilds of the
    coarse space (P was built where the bubbles were), while with full setups they followed the
    exact solves step by step, at the same total cost.
- **Second G100 campaign (P7b, final code; `test/p7/RESULTS.md`)**: AMG `sa` -49% / -56% at 240
  processes (~30k / ~70k cells per process) and -26% / -30% at 960 processes (~7k / ~17k cells, where
  the coarse solve dominates) against exact Cholesky; -20% to -40% against exact factors in single
  precision. The first solve after a rebuild needs the iterations of the exact solves.
- **Reuse policy.** `--reuse auto` works unchanged: it measures what a refactorization and a rebuild
  cost (now with lower medians, see above), so with AMG it decides by itself. A rebuild still
  factorizes Ã_ii exactly for the eigensolver: with AMG this is the largest setup cost.
- **Build.** AMGCL is header-only and bundled; `USE_AMGCL=0` builds without it (`--local-approx amg`
  then stops with a message).
- `test/p7/Allrun_p7b.sh` (G100, 5 nodes, ~35 min) and `Allrun_p7b_960.sh` (20 nodes, ~12 min)
  compare exact factors (double, single) and AMG `sa` / `rs` with the new defaults on the bubbles
  problem at 7k-70k cells per process; `report_p7.py` writes the report and the plots.

## GAMG reference (OpenFOAM)

The target is a preconditioner competitive with the GAMG of OpenFOAM, so before any porting the
OpenFOAM solvers are measured on the same systems as `schwarz2lvl`:

- `openfoam/bubblyPressureFoam` (OpenFOAM-10) assembles the pressure equation of `bubbles3d`,
  −∇·(β_f ∇p) = α with β = 1/ρ, on a uniform blockMesh: α is computed at the cell centres by
  `include/utils/bubble_field.hpp`, the header-only class the generator itself uses (independent of
  the mesh decomposition), β_f by linear interpolation, the flux by an orthogonal snGrad, p = 0 on
  top. The bubble field is compiled with the schwarz2lvl compiler and linked through a C interface
  (`bubbleFieldCapi.h`): compiled by the OpenFOAM toolchain of G100 (icpx oneAPI 2021, `-O3
  -xCORE-AVX512`) it moved 2 of 150 bubbles. The result is the generated matrix and right-hand side
  up to round-off (`-writeMatrix` of a 24³ case against `matgen`: ~1e-15; the G100 jobs stop if not).
  Every solve starts from 0, is timed between two synchronizations of all processes and is followed
  by the true residual ||b − Ax||₂/||b||₂; the OpenFOAM stopping test is on the normalised L1
  residual, `relTol 1e-9` so that the 2-norm reaches ~1e-8 like schwarz2lvl.
- The generator moved its bubble field to `BubbleField`: the matrices are unchanged (bitwise without
  FMA contraction; with `-march` FMA the round-off may differ in the last bit).
- `test/gamg/` (G100): `Allprep_gamg.sh` (meshes, builds and matrix check on part of one node),
  `Allrun_gamg.sh` (5 nodes): two-level Schwarz with AMG / exact local solves, GAMG with DIC /
  GaussSeidel smoothers and PCG + GAMG (optionally GAMG with the coefficient-based `algebraicPair`
  agglomeration and PCG + DIC), on the bubbles problems of P7b (192³ and 256³, 240 processes);
  `report_gamg.py`. Second campaign (run_gamg_23038420, both codes to 1e-8): GAMG with the DIC
  smoother was the fastest OpenFOAM variant (139 / 271 V-cycles per solve at 192³ / 256³: the
  geometric `faceAreaPair` agglomeration ignores the density jumps); the two-level Schwarz with AMG
  local solves needed 51 iterations at both sizes and cost 0.51× / 0.23× of it per solve, setup
  and rebuilds included.
- **OpenFOAM stopping test** (`--stop-norm openfoam`): OpenFOAM stops on the normalised residual
  Σ|b − Ax| / F, F = Σ|Ax₀ − x̄A1| + Σ|b − x̄A1| (x̄ = mean of x₀), at `relTol` times the initial
  one. GMRES evaluates Σ|r_j| of its own residual r_j = g_{j+1} V_{j+1} Qᵀe_{j+1} at every iteration
  (one more pass over the basis, timer `solve.stopping_test`), so both codes stop at the same
  accuracy in the same norm (`test_stopping`). The time loop then also feeds the reuse policy with
  the reduction in that norm.
- **Warm-up of the direct solver.** On G100 the first factorization of the first schwarz2lvl run of a
  job took ~15 s instead of ~1 s on every process (cold shared libraries / kernels; later runs of the
  same job did not show it). The initial setup of that run was inflated (16.3 s instead of 1.5 s),
  and with it the rebuild cost of the reuse policy, which then waited too long before rebuilding (the
  sawtooth of the iterations at 192³). `runTimeLoop` now factorizes a 20³ Laplacian with the local
  backend before the time loop (`warmUpLocalSolver`, not timed, printed), and the scaling jobs start
  with an unmeasured small run.
- **Scaling campaign** (`test/gamg/Allrun_scaling.sh`, `submit_scaling.sh`): GAMG (DIC) against the
  two-level Schwarz (AMG local solves) at relTol 1e-2 of the OpenFOAM residual for both, 192³ /
  256³ / 320³ on 240 / 336 / 432 processes, time of every phase; `report_scaling.py` merges the jobs
  and plots strong and size scaling, the ratio of the costs, the phases and the iterations.
  Result (`test/gamg/RESULTS.md`): 1.8-2.7× cheaper per solve than GAMG on all 9 points, setup
  included; a GMRES iteration costs as much as a V-cycle, GAMG needs 2.0-2.9× more of them (32 → 69
  V-cycles from 192³ to 320³, Schwarz 15 → 25 iterations); parallel efficiency 240 → 432 processes
  0.85-1.18 against 0.56-0.96. Orthogonalization and stopping test of GMRES are 12-20% of the cost,
  the setup and the rebuilds of the coarse space ~30%.

## BiCGStab

In the scaling campaign the orthogonalization of GMRES (CGS2, O(j) passes over the Krylov basis at
iteration j) and the stopping test (one more pass to form the residual for Σ|r|) were 23-28% of the
time of the iterations, growing with the mesh (more iterations, longer vectors). A Krylov method with a
fixed amount of vector work per iteration removes most of it. CG would need a symmetric preconditioner,
i.e. ASM instead of RAS: on the bubbly matrices ASM needs ~1.5× the iterations of RAS, more than CG
saves per iteration. BiCGStab accepts the nonsymmetric RAS/deflated preconditioner as it is.

- **Algorithm** (`BiCgStab`, `--krylov bicgstab`): right-preconditioned BiCGStab exactly as
  `PBiCGStab` of OpenFOAM: shadow residual r̂ = r₀, y = M⁻¹p, z = M⁻¹s, ω = (t, s)/(t, t),
  x += αy + ωz, stopping test on the intermediate residual s (half step) and on r. A port could
  therefore use the PBiCGStab of OpenFOAM with the two-level Schwarz as an `lduMatrix::preconditioner`.
- **Cost per step**: 2 applications of M⁻¹, 2 SpMV, ~12 vector operations and 4 reductions (OpenFOAM: 6;
  here (t, t) with (t, s), and the stopping norm of r with (r̂, r) of the next step). GMRES: per
  iteration 1 application, 1 SpMV, ~4j vector operations (CGS2), 3 reductions, plus with the OpenFOAM
  test one pass over the basis and one reduction.
- **Counting**: `iterations` = applications of the preconditioner (two per BiCGStab step, one if the
  test is met at the half step), the unit of a GMRES iteration, so iterations, time per iteration and the
  reuse policy compare directly; `--maxit` counts applications too.
- **Robustness**: restart with r̂ = r when (r̂, r) ≈ 0 or ω = 0; at convergence the true residual is
  recomputed (10% slack as GMRES), otherwise BiCGStab restarts from it; (r̂, Ay) = 0 is a breakdown.
- **Expected**: BiCGStab does not minimise the residual, so it needs at least as many applications as
  unrestarted GMRES. Locally (bubbles3d 48³, 8 processes, AMG local solves, OpenFOAM test): 8 / 12 / 23
  applications at relTol 1e-2 / 1e-4 / 1e-8 against 7 / 11 / 20 GMRES iterations (1.09-1.15×). With the
  timers of the scaling campaign one application costs ~0.85× (192³) to ~0.75× (320³) of a GMRES
  iteration, so BiCGStab gains if it needs less than ~1.2× (192³) to ~1.3× (320³) the applications.
  `test/gamg/Allrun_krylov.sh` measures it on G100 (240 processes, the scaling matrices, relTol 1e-2),
  `report_krylov.py` splits the result into applications × time per application.

**Limitations.** Real matrices only; one subdomain per MPI process; the global matrix lives on rank 0 during the
decomposition; local indices are 32-bit and the MUMPS / MKL interfaces require n0 < 2^31.
