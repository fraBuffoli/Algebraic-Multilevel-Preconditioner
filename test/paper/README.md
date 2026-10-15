# Reproduction of the paper experiments on G100

Experiments of Section 4 of Al Daas, Jolivet, Rees, *Efficient algebraic two-level Schwarz
preconditioner for sparse matrices* (SISC 2023), within 5 nodes (240 MPI processes, one
subdomain per process).

```bash
cd test/paper
./get_matrices.sh                    # once, on a login node: SuiteSparse matrices -> ../../matrices
sbatch Allrun.sh                     # one job, 5 nodes: build + all the runs in series (~1-1.5 h)
python3 plot_paper.py run_<jobid>    # report.md, summary.csv, phases.csv, plots/
python3 plot_paper.py run_<jobid> --ref run_22838900   # + before -> after for the runs with the same name
```

| Step in `Allrun.sh` | Paper | Setting here |
|---|---|---|
| `RUN_CHECK` | - | build + MPI unit tests on 16 processes |
| `RUN_FIG4` | Fig. 4: strong scaling, 3D (4.1), nu = 1e-2, n = 8.1e6, N = 32...512 | convdiff3d 200^3 = 8.0e6, N = 30, 60, 120, 240 |
| `RUN_TAB2` | Table 2: nu = 1...1e-4, N = 1024 (2D) / 4096 (3D) | same subdomain size (~6k 2D, ~2k 3D): 1200^2 and 78^3 on N = 240 |
| `RUN_TAB1` | Table 1: 19 SuiteSparse matrices, N = 256, tau = 0.6, nev = 300 | 11 real matrices, N = 240, tau = 0.6 and 0.9 (`TAB1_TAUS`); one-level RAS as reference |
| `RUN_FIG2` | Fig. 2: G3_circuit, tau = 0.02...0.9 | same, N = 240, plus one-level ASM; the tau of Table 1 are taken from there |

All runs: GMRES(30) with right preconditioning, rtol 1e-8, zero initial guess; random right-hand side
for the matrices, boundary-condition right-hand side for (4.1). Steps can be switched off at the top
of `Allrun.sh` (e.g. to split the budget over two jobs); `RUN_TIMEOUT` stops a single run without
losing the job (on Table 1, a timeout skips the larger tau of the same matrix). Each run leaves
`log.<run>` (with the full timing table: min/avg/max over the ranks of every phase), `csv/<run>.csv`
and `hist/<run>.txt` in `run_<jobid>/`; `status.txt` lists the outcome and wall time of every run.
The binary is built inside `run_<jobid>/` with `USE_CPARDISO=1`.

Compared with run 22838900 (same experiments, A_00 on rank 0, nev = 300 always computed):

- A_00 is factorized by MKL `cluster_sparse_solver` on 48 processes (`COARSE`; `--coarse-solver root`
  gives the old behaviour);
- the local eigensolver computes k = 32, 64, ... eigenpairs until the smallest is below 1/tau
  (`--nev-start 32`, default): same selected vectors, so n0 and iterations must not change, but the
  cost of Arnoldi follows the number of selected vectors instead of nev = 300;
- Arnoldi is restarted after a Schur failure (fig4_N240 of run 22838900);
- Table 1 also at tau = 0.9: the G3_circuit row of the paper's Table 1 (n0 = 71,385, 8 it.) is the
  tau = 0.9 point of its own Fig. 2, which run 22838900 reproduced with the same n0/N (278.8);
- circuit5M_dc runs with `--verbose 2`: one line per rank when its eigensolver ends
  (`grep "eigensolver done" log.tab1_circuit5M_dc_2L`), to find the ranks that kept the setup
  running for more than 20 min in run 22838900.

Compared with run 22843541 (current `Allrun.sh`, `--ref run_22843541`), after the performance
campaign of run 22861093 (`Allrun_perf.sh`, below):

- PARDISO without iterative refinement (local solver and `cpardiso`), kernel of Ã_ii from
  multi-RHS solves: same n0 and iterations, faster Arnoldi, kernel detection and solve;
- `COARSE` adds `--coarse-root-below 10000`: A_00 goes to rank 0 when n0 < 10000 (crossover
  of the root / cp48 calibration; it also avoids the cluster_sparse_solver crash of finan512
  at tau = 0.6, n0 = 109);
- the logs split the preconditioner application (`solve.preconditioner.coarse.*`,
  `.spmv`, `.local.{halo,lu}`) and report the perturbed pivots of A_ii and Ã_ii + σI;
  `git_status.txt` lists the files that differ from the commit in `log.git`.

`--ref` adds a before -> after table (iterations, n0, Arnoldi, A_00 factorization, setup, solve, total)
for the runs with the same name in both directories.

Differences from the paper to keep in mind when comparing: N = 240 instead of 256 (compare n0/N);
METIS instead of ParMETIS; finite volumes with upwind instead of SUPG finite elements for (4.1);
MKL cluster_sparse_solver instead of MUMPS for the coarse problem (no MUMPS on G100).

## Second campaign: coarse solvers and multiplicity partition of unity

```bash
sbatch Allrun_coarse.sh                     # one job, 5 nodes, ~3 h
python3 plot_coarse.py run_coarse_<jobid>   # report.md, summary.csv, phases.csv, plots/
```

The binary is built with `USE_CPARDISO=1` inside `run_coarse_<jobid>/`. On the experiments above it compares

- coarse solvers (`COARSE`): `root` (A_00 gathered and factorized on rank 0) and `cpK` = MKL
  `cluster_sparse_solver` on K processes, one every 240/K ranks (default K = 240, 48, 10);
- partitions of unity (`POUS`): Boolean and multiplicity;
- threshold tau = 0.6 (paper) and 0.9 on Table 1.

| Step | Runs |
|---|---|
| `RUN_CHECK` | unit tests (every coarse solver vs a dense reference), then laplace3d / convdiff3d 100^3 on 240 processes with every PoU and coarse solver: same n0 and iterations expected; if a `cpK` run fails the job stops |
| `RUN_EIG` | Table 1 matrices, Boolean, tau = 0.6, root, with `--nev-start 0` (nev = 300 always computed): reference for the default adaptive k of all the other runs; same n0 and iterations expected |
| `RUN_TAB1` | Table 1 matrices, nev = 300: Boolean tau = 0.6 and 0.9 with every coarse solver; multiplicity tau = 0.6 with `CP`; one-level RAS for each PoU. circuit5M_dc is left out (two-level setup > 20 min in run 22838900) |
| `RUN_FIG2` | G3_circuit, tau = 0.02 ... 0.9, both PoU, coarse solver `CP` (cp48) |
| `RUN_TAB2` | convection-diffusion nu = 1 ... 1e-4 (2D 1200^2, 3D 78^3), both PoU, two-level only |
| `RUN_FIG4` | strong scaling 3D 200^3, N = 30 ... 240, both PoU |
| `RUN_DIAG` | circuit5M_dc with `--verbose 2`; the ranks missing from `grep "eigensolver done" log.diag_*` are the slow ones |

The log of every two-level run reports the coarse solver used, e.g.
`coarse solver cpardiso(48 procs, mtype -2, nnz(LU) ..., perturbed pivots 0)`, and the timers
`setup.coarse_factorization.{exchange,analysis,numeric}` split the distributed factorization;
`setup.eigensolver.arnoldi.operator` is the part of Arnoldi spent applying the operator (sparse solves).

## Third campaign: performance of setup and solve

```bash
sbatch Allrun_perf.sh                                        # one job, 5 nodes, ~30 min
python3 plot_paper.py run_perf_<jobid> --ref run_22843541    # before -> after, same run names
cat run_perf_<jobid>/cal_summary.txt                         # root vs cpardiso -> --coarse-root-below
```

Changes measured (same preconditioner: n0 and iterations must not change):

- PARDISO without iterative refinement, local solver and `cpardiso` (`iparm[7] = 0`; Eigen sets 2);
- kernel detection of Ã_ii with one multi-RHS solve per inverse iteration step (2 steps in run 22861093,
  3 again afterwards: with 2, one vector of nxp1 crossed kernel_tol and n0 grew by 1);
- timers of the preconditioner application: `solve.preconditioner.coarse.{restrict,solve,prolong}`,
  `solve.preconditioner.spmv`, `solve.preconditioner.local.{halo,lu}`;
- `--coarse-root-below N`: root coarse solver when n0 < N.

| Step | Runs |
|---|---|
| `RUN_CHECK` | build + unit tests on 16 processes |
| `RUN_REF` | fig4_N30 ... N240, tab2_{2d,3d}_nu1e-2_2L, tab1_{G3_circuit,memchip,CoupCons3D,nxp1}_2L (tau 0.6) and G3_circuit at tau 0.9, with the names and options of run_22843541 |
| `RUN_CAL` | root vs cp48 on the same problems with n0 from ~0.7k to ~25k; `cal_summary.txt`: iterations, A_00 factorization and coarse solve per iteration for both |
| `RUN_THR` | finan512, tau = 0.6 (n0 = 109, crashed inside cluster_sparse_solver) with `--coarse-root-below THR` |
