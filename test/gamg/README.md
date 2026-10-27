# GAMG reference — OpenFOAM linear solvers and two-level Schwarz on the same matrices (G100)

Results of the campaigns (`run_gamg_23036378`, `run_gamg_23038420`, scaling, Krylov, rebuilds, coarse space
size): [`RESULTS.md`](RESULTS.md). Latest: [Krylov campaign](#krylov-campaign-allrun_krylovsh) (GMRES vs
BiCGStab with the same preconditioner), [rebuilds](#rebuilds-run_rebuild_np240_23093509) (cheaper rebuilds
of the coarse space), [coarse space size](#coarse-space-size-campaign-allrun_nevsh) (`--nev` 8 / 12 / 16:
8 stays). Setting to port: BiCGStab, `--tau 0.3 --nev 8 --eig-tol 1e-2 --eig-ncv 12`, AMG local solves.

**Idea.** The goal is a preconditioner competitive with the GAMG of OpenFOAM. This campaign measures
where we stand, before any porting: OpenFOAM solves the pressure matrices of the `bubbles3d` problem
of the previous campaigns with its own solvers, and `schwarz2lvl` solves the same matrices.

- `openfoam/bubblyPressureFoam` (OpenFOAM-10 application, ~320 lines): at every step it computes the
  gas fraction alpha at the cell centres with `include/utils/bubble_field.hpp`, the code the generator
  itself uses, assembles `-laplacian(beta_f, p) = alpha` (beta = 1/rho, linear interpolation,
  orthogonal snGrad, p = 0 on top, zero flux elsewhere) and solves it with the controls
  `solvers/<name>` of `system/fvSolution`. On the uniform mesh this is the matrix of
  `schwarz2lvl --generate bubbles3d --rhs generated`.
- **Same matrices, checked.** The bubble field of the application (`bubbleFieldCapi.cpp`, C
  interface) is compiled with the schwarz2lvl compiler and flags and linked into the application:
  compiled by the OpenFOAM toolchain of G100 (icpx oneAPI 2021, `-O3 -xCORE-AVX512`) the same header
  moved 2 of the 150 bubbles (first campaign, run_gamg_23036378), while g++ and clang, with or without
  fast math, agree to 1e-15. Both jobs compare the OpenFOAM matrix of a 24³ case (`-writeMatrix`) with
  `matgen` and stop if they differ.
- **Same accuracy.** Every solve starts from 0. schwarz2lvl stops at ||b − Ax||₂/||b||₂ = 1e-8.
  OpenFOAM tests its normalised L1 residual (from x = 0: Σ|b − Ax| / Σ|b|): with `relTol 1e-8` GAMG
  stopped at up to 1.2e-7 in the 2-norm, so `fvSolution` uses `relTol 1e-9` (2-norm ~1e-8). Both
  codes report the true 2-norm residual of every solve.
- **Same timing.** Every solve is timed between two synchronizations of all processes; matrix
  assembly is not timed in either code. `schwarz2lvl` also pays its setup and the updates of the
  reuse policy (`--reuse auto`); GAMG agglomerates its coarse matrices in every solve (the
  agglomeration itself is cached, the mesh does not change).
- Problems of P7b: 150 bubbles, density ratio 1000, rising 0.5 cells per step; 192³ (20 steps) and
  256³ (16 steps), 3 solves per step, 240 processes (~30k / ~70k cells per process).

| variant | solver | run |
|---|---|---|
| `amg` | two-level Schwarz, P7b setting: AMG V-cycle local solves, `--tau 0.3 --nev 8 --eig-tol 1e-4` | both sizes |
| `exact` | the same with exact Cholesky local solves | both sizes |
| `GAMG_DIC` | GAMG, DIC smoother (fastest OpenFOAM variant of the first campaign) | both sizes |
| `GAMG_GS` | GAMG, GaussSeidel smoother, `faceAreaPair` agglomeration: the usual p / p_rgh setup (baseline) | both sizes |
| `PCG_GAMG` | CG preconditioned by 2 GAMG V-cycles (DICGaussSeidel) | 192³ |
| `GAMG_ALG` | GAMG, `algebraicPair` agglomeration (cells paired by matrix coefficients) | off (`OF_A`) |
| `PCG_DIC` | CG + incomplete Cholesky local to each process | off (`OF_A`) |

`GAMG_ALG` and `PCG_DIC` were slower than `GAMG_GS` / `GAMG_DIC` in the first campaign (229 / 2019
iterations per solve at 192³); add them with e.g. `OF_A="GAMG_DIC GAMG_GS PCG_GAMG GAMG_ALG"`.

```bash
cd test/gamg
jid=$(sbatch --parsable Allprep_gamg.sh)            # part of 1 node: meshes (kept if present), builds, matrix check (~2 min)
sbatch --dependency=afterok:$jid Allrun_gamg.sh     # 5 nodes = 240 processes, ~45 min (limit 70), ~180 core-hours
python3 report_gamg.py run_gamg_<jobid>             # -> report.md + plots/gamg_cost.png, plots/gamg_solves.png
```

`Allprep_gamg.sh` writes the decomposed meshes to `cases/bub<N>_np<NP>` (scotch; hierarchical if scotch
fails; meshes already prepared with the same template are kept), builds the application and `matgen`
and runs the matrix check: if anything fails the job fails and the dependent `Allrun_gamg.sh` stays
pending (`scancel` it). Without the prep job `Allrun_gamg.sh` prepares missing meshes itself, with
~10 minutes of idle cores. Delete `cases/` when done (~3 GB).

Local check of the scripts (OpenFOAM-10 environment sourced, any MPI, tiny sizes):
`SMOKE=1 NP=4 ./Allprep_gamg.sh` and `SMOKE=1 NP=4 ./Allrun_gamg.sh`.

## Scaling campaign (`Allrun_scaling.sh`)

Second step, after `run_gamg_23038420` (GAMG_DIC fastest OpenFOAM variant; Schwarz with AMG local solves
0.51× / 0.23× of its cost per solve at 192³ / 256³, both to 1e-8): one against one, at the accuracy of a
pressure solve in a real time step, on a larger mesh and on more processes.

- **Solvers**: `GAMG_DIC` (OpenFOAM) against `amg` (two-level Schwarz, P7b setting:
  `--tau 0.3 --nev 8 --eig-tol 1e-4 --local-approx amg`, reuse policy `auto`).
- **Same stopping test**: the normalised residual of OpenFOAM, relTol **1e-2** for both (OpenFOAM:
  `relTol 1e-2`, `tolerance 0`; schwarz2lvl: `--stop-norm openfoam --rtol 1e-2`). From x = 0 both stop
  at Σ|b − Ax| ≤ 1e-2 Σ|b|. `RTOL=...` changes it for both.
- **Grid**: 192³, 256³, 320³ (7.1M, 16.8M, 32.8M cells) × 240, 336, 432 processes (one job of 5, 7, 9
  nodes each), 16 steps × 3 correctors for every size (same amortization of the setup).
- **Phases** (every solve, between two synchronizations): Schwarz initial setup / updates (rebuild,
  refactorization, reuse) / GMRES iterations, the iterations split into local solves, coarse correction and
  Krylov with the timers of the log; GAMG construction of the solver (agglomeration in the first solve,
  coarse matrices in every solve) / V-cycles. Matrix assembly and generation are not compared.
- **Anomalies of the previous campaign fixed**: the first factorization of the first schwarz2lvl run of a
  job took ~15 s (cold start) → warm-up of the direct solver before the time loop of every run, plus one
  small unmeasured run at the start of every job; the sawtooth of the iterations at 192³ (the policy
  used the inflated initial setup as rebuild cost) goes with it. The report checks both
  (*initial setup / median rebuild* ≈ 1).

```bash
cd test/gamg
./submit_scaling.sh          # Allprep_scaling.sh (1 node, ~15-20 min) + Allrun_scaling.sh on 5, 7, 9 nodes after it
                             # (~12-15 min each); ~200-250 core-hours in all
python3 report_scaling.py .  # every run_scaling_np*_<jobid> here -> scaling_report/report.md, summary.csv, plots/
```

`Allprep_scaling.sh` makes one serial mesh per size (`cases/base<N>`) and decomposes it for every process
count (`cases/bub<N>_np<NP>`, linked base mesh; cases already there are kept, including those of
`Allprep_gamg.sh`), and builds and checks the binaries once (`cases/bin_<sources md5>`: test_bubbles + matrix
check); `Allrun_scaling.sh` uses them if the sources did not change, otherwise it builds and checks itself.
Options: `NODES="5 7 9"`, `SERIAL=1` (one job after the other), `PREP=0` (skip the preparation) for
`submit_scaling.sh`; `SIZES`, `STEPS`, `CORR`, `RTOL`, `OF_SOLVER` for `Allrun_scaling.sh`. A failed
preparation leaves the run jobs pending (`scancel` them). Delete `cases/` when done (~30 GB).
Local check: `SMOKE=1 NPS="2 4" ./Allprep_scaling.sh`, then `SMOKE=1 NP=4 ./Allrun_scaling.sh`.

Plots of `report_scaling.py`: `scaling_strong.png` (cost per solve and parallel efficiency vs processes),
`scaling_size.png` (cost per solve vs cells), `scaling_ratio.png` (GAMG / Schwarz), `scaling_phases.png`
(seconds per solve by phase), `scaling_iterations.png` (iterations of every solve).

## Krylov campaign (`Allrun_krylov.sh`)

Third step: the same preconditioner (RAS + deflated, AMG local solves) inside GMRES(30) and inside
BiCGStab (see [BiCGStab](../../README.md#bicgstab)). In the scaling campaign orthogonalization and
stopping test of GMRES were 23-28% of the time of the iterations; BiCGStab has a fixed amount of vector
work per application of the preconditioner but needs more applications. The campaign measures the net
effect, and whether OpenFOAM's PBiCGStab can host the preconditioner without losing performance.

- **Variants** (`VARIANTS`, default `amg bicgstab`): `amg` = GMRES(30) + RAS/deflated (the reference,
  same setting as the scaling campaign), `bicgstab` = BiCGStab + RAS/deflated.
- **Problem and test**: the matrices of the scaling campaign (192³, 256³, 320³, 16 steps × 3
  correctors), OpenFOAM stopping test with relTol 1e-2 (`RTOL=1e-8` for the asymptotic regime), 240
  processes (5 nodes).
- **GAMG** is not rerun (`OF_SOLVER=none`): `report_krylov.py --gamg run_scaling_np240_<jobid>` takes it,
  and the GMRES run of the scaling campaign as a check of run-to-run variation, from the scaling
  campaign. `OF_SOLVER=GAMG_DIC` runs it in the same job (OpenFOAM cases needed).
- Nothing to compile beforehand: the job builds schwarz2lvl on the compute node, once per source
  version (`cases/bin_s2l_<md5>`), with the modules of the scaling campaign (`autoload openfoam/10`
  for the Intel compilers and Intel MPI, `metis`, `intel-oneapi-mkl`; no OpenFOAM build) and checks it
  with `test_bicgstab` and `test_stopping` on 4 processes. A failed build or test stops the job with
  `STOP: ...` in `allrun_krylov.out`.

```bash
cd test/gamg
sbatch Allrun_krylov.sh                  # 5 nodes, ~25 min with the build (2 variants x 3 sizes)
python3 report_krylov.py run_krylov_np240_r1e-2_<jobid> --gamg run_scaling_np240_23040226
                                         # -> krylov_report/report.md, summary.csv, plots/
python3 plot_krylov.py run_krylov_np240_r1e-2_<jobid> --gamg run_scaling_np240_23040226
                                         # -> run_krylov_np240_r1e-2_<jobid>/plots/ (--gamg required)
```

`report.md`: iterations (= applications of the preconditioner) per solve, time per application split
into local solves / coarse correction / SpMV / orthogonalization or vector operations / stopping test,
cost per solve against GMRES and GAMG, and the ratio BiCGStab / GMRES split into applications × time per
application. `plot_krylov.py` writes the plots into the run directory itself (`<run>/plots/`):
`cost_per_solve_np240.png` (phases, GAMG bar), `time_per_iteration_np240.png` (split of one
application) and `solves_np240.png` (iterations and time of every solve, rebuilds marked, GAMG
V-cycles). Local check: `SMOKE=1 NP=4 ./Allrun_krylov.sh`. With `RTOL=1e-8` the iterations triple:
`sbatch --time=01:00:00`.

**Result** (`run_krylov_np240_r1e-2_23091304`, [RESULTS.md](RESULTS.md) section 5): BiCGStab costs
1.02 / 0.89 / 0.82× GMRES per solve at 192³ / 256³ / 320³ (1.04-1.20× the applications, each
0.72-0.85× as expensive; same rebuilds), 1.8 / 2.2 / 2.3× cheaper than GAMG.

## Rebuilds (`run_rebuild_np240_23093509`)

Fourth step: cheaper rebuilds of the coarse space (BiCGStab, 256³ / 320³, 24 steps, 240 processes),
run with a prototype that is not kept in the code (reuse of the factorization of Ã_ii between
rebuilds). Result ([RESULTS.md](RESULTS.md) section 6): the useful part needs no code,
`--eig-tol 1e-2 --eig-ncv 12` (rebuild −22-31%, initial setup −18%, same coarse space); the reuse of
the factorization saved only 14% of a rebuild. The run directory keeps logs, CSV and plots.

## Coarse space size campaign (`Allrun_nev.sh`)

Fifth step, the last before a port: `--nev 8 / 12 / 16` at the practical tolerance. The cap 8 was chosen
to 1e-8 with exact local solves, when coarse solve and eigensolver dominated; with AMG local solves at
relTol 1e-2 it is reached in every subdomain (τ = 0.3 never selects fewer), and the iterations right
after a rebuild still grow from 192³ to 320³ (16 → 16 → 18-19).

- **Question**: does a larger coarse space buy back mesh independence and fewer (or cheaper) aged
  iterations and rebuilds, for its larger setup (Arnoldi applications ~ nev) and coarse solve
  (n0 = nev × processes, on rank 0 below 10⁴: 6912 at 432 processes with nev 16)?
- **Variants**: `nev8`, `nev12`, `nev16`, all BiCGStab + RAS/deflated, AMG local solves, τ 0.3, the
  looser Arnoldi of section 6 of RESULTS.md (`--eig-tol 1e-2`, `--eig-ncv` = 1.5 nev), reuse `auto`.
  Locally (64³, 8 processes, one bubble per subdomain) m_i = nev everywhere, 23 / 35 / 45 operator
  applications, 25 / 23 / 21 iterations at relTol 1e-8 for a fresh coarse space.
- **Grid**: 192³, 256³, 320³, 16 steps × 3 correctors (as the scaling campaign), relTol 1e-2, one job per
  process count: 240 and 432 (`NODES="5 9"`; `NODES="5 7 9"` adds 336). GAMG from the scaling campaign.
  Run: 240 processes only (`run_nev_np240_23094441`); the 432 job was cancelled (see the result).

```bash
cd test/gamg
./submit_nev.sh                          # 2 jobs (5 and 9 nodes), the second after the first; ~30 min each
python3 report_nev.py run_nev_np240_<jobid> run_nev_np432_<jobid> \
    --gamg run_scaling_np240_23040226 --gamg run_scaling_np432_23040228
                                         # -> report.md, summary.csv, plots/ inside every run directory
```

`report.md`: n0, m_i, operator applications, iterations after every rebuild and on average, GAMG
V-cycles; initial setup, rebuilds and their cost, coarse correction per iteration, cost per solve (also
with machine stalls removed: iterations at the median time per iteration of the run), ratios to nev 8
and to GAMG. Plots: `nev_iterations_np<NP>.png` (iterations vs mesh, fresh and average: mesh
dependence), `nev_cost_np<NP>.png` (cost per solve by phase, GAMG), `nev_solves_np<NP>.png` (every
solve, rebuilds marked). The first job builds and checks the binaries, the second reuses them. Local
check: `SMOKE=1 NP=4 ./Allrun_nev.sh`. Report of the run that was made:
`python3 report_nev.py run_nev_np240_23094441 --gamg run_scaling_np240_23040226`.

**Result** (`run_nev_np240_23094441`, [RESULTS.md](RESULTS.md) section 7): nev 12 / 16 take 8-22% fewer
applications, but each costs 6-28% more (the coarse correction grows with nev: products with the local
basis, coarse solve on rank 0) and the setup grows by 20-61% (Arnoldi, coarse matrix at every
refactorization): 1.04-1.06× (nev 12) and 1.15-1.20× (nev 16) the cost per solve of nev 8. The gain
shrinks with the mesh (right after a rebuild at 320³: 16 / 15 / 15), so no mesh independence is bought
back, and the coarse space ages at the same rate (one rebuild in every run). `--nev 8` stays. Not run at
432 processes: smaller subdomains, a doubled coarse problem on rank 0 (n0 = 6912) and a coarse correction
that already weighs more there would make the extra cost per application larger than at 240.

## What to look at in `report.md` (`Allrun_gamg.sh`)

1. **Checks**: `test_bubbles` passed, matrix check `OK`, all runs `ok`, every solve converged with true
   residual ~1e-8 for both codes.
2. **Cost per solve** and the line *Fastest OpenFOAM variant / best schwarz2lvl*: *median* is the
   steady state (no setup), *cost/solve* includes the setup and the rebuilds of `schwarz2lvl`.
3. **Iterations of GAMG**: the faceAreaPair agglomeration ignores the density jumps; with 150 bubbles
   at ratio 1000 GAMG needed 120-190 V-cycles at 192³ and 230-330 at 256³ in the first campaign.
4. `plots/gamg_solves.png`: time of every solve at 256³ (setup and rebuild spikes of `schwarz2lvl`).

**Caveats.** OpenFOAM 10 (the G100 module) while the target is OpenFOAM 14: GAMG is essentially the
same. Solves from 0 to 1e-8 measure the asymptotic convergence; in a real time loop OpenFOAM starts
from the previous solution with loose tolerances (`relTol 0.05`, final `tolerance 1e-7`), where the
setup cost of schwarz2lvl weighs more (`-warmStart` of the application exists, schwarz2lvl starts from
0). The decompositions differ (scotch / METIS). A port would add the conversion of the OpenFOAM
matrix to CSR at every solve (not measured here).
