# schwarz2lvl — Algebraic Two-Level Schwarz Preconditioner

C++17 / MPI implementation of the fully algebraic overlapping two-level Schwarz preconditioner of

> H. Al Daas, P. Jolivet, T. Rees, *Efficient Algebraic Two-Level Schwarz Preconditioner for Sparse Matrices*,
> SIAM J. Sci. Comput. 45(3), A1199–A1213, 2023. doi:10.1137/22M1469833

The second level is a spectral coarse space built from the matrix alone: every subdomain solves a local
generalized eigenproblem involving the lumped matrix Ã_ii (Definition 3.1), which is assembled in O(n_i)
operations from the distance-one overlap. One MPI process owns one subdomain; the preconditioner is used
inside a right-preconditioned restarted GMRES. Real arithmetic only.

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
./bin/schwarz2lvl --help                                                                 # full option list
```

Options are given as `--key value` or `--key=value`. Defaults reproduce the setup of Section 4 of the paper
(zero initial guess, random right-hand side, τ = 0.6, nev = 300, GMRES(30), rtol = 1e-8).

| Group | Options |
|---|---|
| Input | `--matrix FILE` (Matrix Market coordinate, real/integer/pattern, general/symmetric/skew) **or** `--generate laplace2d\|laplace3d\|convdiff2d\|convdiff3d\|coupled2d\|coupled3d` with `--nx --ny --nz --nu --ncomp`; `--rhs random\|ones\|generated\|FILE`; `--block-size N` (unknowns per cell) |
| Partition | `--metis-objective cut\|vol`, `--metis-vertex-weights 0\|1` |
| Preconditioner | `--one-level ras\|asm`, `--coarse deflated\|additive\|none`, `--pou boolean\|multiplicity`, `--tau`, `--nev` |
| Local eigensolver | `--eig-tol --eig-maxit --eig-ncv`, `--shift-rel`, `--kernel-tol`, `--kernel-probe`, `--dense-threshold`, `--orth-tol` |
| Direct solvers | `--local-solver auto\|sparselu\|umfpack\|pardiso`, `--coarse-solver auto\|mumps\|cpardiso\|root`, `--coarse-procs N` |
| GMRES | `--restart`, `--rtol`, `--maxit` |
| Output | `--verbose 0\|1\|2`, `--csv FILE`, `--history FILE`, `--write-solution FILE` |

`auto` picks the best backend compiled in (local: pardiso > umfpack > sparselu; coarse: mumps > cpardiso > root).

**Output.** Rank 0 prints the configuration, min/avg/max subdomain sizes (n_I, n_Γ, neighbours, nnz(A_ii)), coarse
space statistics (n0, m_i, kernel dimensions, Arnoldi convergence), GMRES iterations with the GMRES and true relative
residuals, and a min/avg/max timing breakdown of every phase. `--csv` appends one line per run (problem, n, nnz, np,
options, n0, iterations, residual, setup / eigensolver / coarse assembly / coarse factorization / solve times),
which is the format intended for strong/weak scaling studies. `--history` writes the residual at every iteration,
`--write-solution` the solution in the original ordering. Exit code: 0 converged, 2 not converged, 1 invalid input.

**matgen** writes the built-in problems in Matrix Market format (e.g. to compare with PETSc/PCHPDDM):

```bash
./bin/matgen convdiff3d 40 nu=0.01 out=cd3d_40      # -> cd3d_40.mtx, cd3d_40_rhs.mtx
```

**Tests** compare the distributed implementation with sequential references built from the global matrix;
run them with several process counts:

```bash
mpirun -np 4 ./bin/test_subdomain        # overlap, halo, A_ii, lumping sums, partition of unity, SpMV
mpirun -np 4 ./bin/test_preconditioner   # sparse vs dense eigensolver, RAS/ASM x deflated/additive, GMRES
```

`matrices/` contains small SuiteSparse test matrices (sherman4, nos5, e05r0400, fidapm05/10, garon1, graham1,
lns_3937, cage7/10, windtunnel_evap2d).

## Code structure

```
include/<module>/*.hpp, src/<module>/*.cpp      one class per header/source pair, namespace schwarz2lvl::<ns>
├── core/            (core)    Scalar, LocalIndex (32-bit), GlobalIndex (64-bit); SolverConfig (CLI); LocalSystem; MPI helpers
├── partition/       (part)    CsrMatrix, MatrixMarketIO, GraphPartitioner (METIS), DomainDecomposer, Subdomain
├── linalg/          (linalg)  HaloExchange, DistributedMatrix (SpMV), LocalSolver (SparseLU/UMFPACK/PARDISO), global dot/norm
├── preconditioner/  (prec)    Preconditioner, OneLevelSchwarz, local block splitting, LocalEigensolver,
│                              CoarseSpace, CoarseSolver (MUMPS/cpardiso/root), TwoLevelSchwarz
├── solver/          (solv)    Gmres
└── utils/           (utils)   MatrixGenerator, TimerRegistry/ScopedTimer, Log
src/main.cpp, src/matgen.cpp   drivers;   test/   MPI test programs;   external/   Eigen, Spectra
```

Flow of `main.cpp`:

1. **Decomposition** — `DomainDecomposer::decompose`: rank 0 reads or generates the system, partitions G(A + Aᵀ)
   with METIS, renumbers the unknowns and sends each rank its owned rows → `LocalSystem`.
2. **Subdomain** — overlap Ω_i = [Ω_I | Ω_Γ], halo pattern, A_ii, A(Ω_I, Ω_i), lumping sums, partition of unity D_i.
3. **One level** — `OneLevelSchwarz`: exact LU of A_ii; RAS (2.3) or ASM (2.1).
4. **Two level** — `TwoLevelSchwarz`: Ã_ii (Def. 3.1) → `LocalEigensolver` gives Z_i (3.1)–(3.2) → `CoarseSpace`
   builds V_i = orth(D_i Z_i) and A_00 = R_0 A R_0ᵀ → `CoarseSolver` factorizes A_00.
   Deflated (2.4): z = x_0 + M₁⁻¹(r − A x_0), x_0 = R_0ᵀ A_00⁻¹ R_0 r; additive (2.2): z = R_0ᵀ A_00⁻¹ R_0 r + M₁⁻¹ r.
5. **Solve** — `Gmres` with the preconditioner applied on the right.

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
  also the reference in the tests. Selection: the nev largest, kept if |λ| ≥ 1/τ (Section 4).
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
  solve). LDLᵀ (mtype −2) if A is symmetric, LU with scaling and matching otherwise, METIS ordering on the first
  process (the MPI ordering of PARDISO, iparm[1] = 10, hung in tests). With Open MPI the symmetric types crash inside
  MKL on communicators other than MPI_COMM_WORLD, so there symmetric A_00 are factorized as general matrices.
  `root` (gather on rank 0 + sequential LU) is the non-scalable fallback/reference.
- Local: backends selected at run time behind one `LocalSolver` interface. UMFPACK runs with METIS ordering and no
  iterative refinement (faster factorizations and solves on 3D problems); PARDISO is meant to run with one thread
  per rank (`MKL_NUM_THREADS=1`).
  Transposed solves are needed for the left kernel of Ã_ii.
- GMRES: right preconditioning, so the stopping test is on the unpreconditioned residual as in the paper;
  classical Gram–Schmidt applied twice (CGS2), one `MPI_Allreduce` per pass instead of j+1; the true residual is
  recomputed at every restart.
- Test generators: cell-centred finite volumes for Eq. (4.1) with upwind convection, the velocity fields of
  Section 4.2 integrated exactly on the faces (zero row sums, hence singular Ã_ii that exercise the kernel handling),
  heterogeneous diffusivity, and coupled variants with unknowns interleaved per cell as in OpenFOAM coupled solvers.
- The random right-hand side is generated in the original ordering, so it does not depend on the number of processes.

**Limitations.** Real matrices only; one subdomain per MPI process; the global matrix lives on rank 0 during the
decomposition; local indices are 32-bit and the MUMPS / MKL interfaces require n0 < 2³¹.
