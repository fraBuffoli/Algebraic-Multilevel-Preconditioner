# GAMG reference — OpenFOAM linear solvers and two-level Schwarz on the same matrices (G100)

Results of the three campaigns (`run_gamg_23036378`, `run_gamg_23038420`, scaling): [`RESULTS.md`](RESULTS.md).

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
