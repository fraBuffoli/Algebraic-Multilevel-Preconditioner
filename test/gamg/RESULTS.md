# GAMG reference on G100: results

The goal of the project is a preconditioner competitive with the GAMG of OpenFOAM. These campaigns
measure where the two-level Schwarz preconditioner stands **before any porting**: OpenFOAM
(`openfoam/bubblyPressureFoam`) and `schwarz2lvl` solve the same pressure matrices, the bubbles3d
problem of P5/P7b (−∇·(β∇p) = b, β = 1/ρ, 150 rising gas bubbles, density ratio 1000, 0.5 cells of
rise per step, 3 solves per step). Sections 1-7 solve every system from 0 with b = vol·α; section 8
repeats the comparison in the conditions of a PIMPLE loop: the p_rgh source of a VoF solver, every
solve from the previous solution, the last corrector of every step to an absolute tolerance.

| campaign | script | what | outcome |
|:--|:--|:--|:--|
| `run_gamg_23036378` | `Allrun_gamg.sh` (first version) | 5 OpenFOAM variants vs Schwarz `amg` / `exact`, 192³ and 256³, 240 processes, to ~1e-8 | reconnaissance only: OpenFOAM matrices not identical (section 1), timed out at 256³ |
| `run_gamg_23038420` | `Allrun_gamg.sh` | GAMG_DIC, GAMG_GS, PCG_GAMG vs `amg`, `exact`, same problems, **to 1e-8** in the 2-norm | asymptotic comparison, [`report.md`](run_gamg_23038420/report.md) |
| `run_scaling_np240_23040226`, `_np336_23040227`, `_np432_23040228` | `Allrun_scaling.sh` | GAMG_DIC vs `amg`, 192³ / 256³ / 320³ × 240 / 336 / 432 processes, **relTol 1e-2** of the OpenFOAM residual for both | practical tolerance and scaling, [`report.md`](scaling_report/report.md) |
| `run_krylov_np240_r1e-2_23091304` | `Allrun_krylov.sh` | the same Schwarz preconditioner inside GMRES(30) and inside **BiCGStab**, 192³ / 256³ / 320³, 240 processes, relTol 1e-2 | Krylov method (section 5) |
| `run_rebuild_np240_23093509` | prototype script (not kept) | cheaper rebuilds of the coarse space: reuse of the factorization of Ã_ii, looser Arnoldi; BiCGStab, 256³ / 320³, 24 steps, 240 processes, relTol 1e-2 | rebuilds (section 6) |
| `run_nev_np240_23094441` | `Allrun_nev.sh` (not in the repository) | size of the coarse space: `--nev 8 / 12 / 16` with the looser Arnoldi of section 6; BiCGStab, 192³ / 256³ / 320³, 240 processes, relTol 1e-2 | coarse space size (section 7) |
| `run_warm_np240_23097350`, `run_warm_np432_23097354` | `Allrun_warm.sh` | PIMPLE-like loop: p_rgh source, warm start, correctors to relTol 0.05 or tolerance 1e-7, last corrector to tolerance 1e-7; two-level Schwarz, one-level RAS, GAMG_DIC (+ PCG_GAMG for the last corrector at 192³), 192³ / 256³ / 320³, 240 and 432 processes | warm start (section 8), [`report.md`](warm_report/report.md) |

Unless stated otherwise, Schwarz = `--tau 0.3 --nev 8 --eig-tol 1e-4` (coarse space of P5), `--local-approx amg` (one AMG
V-cycle per local solve, P7b; `exact`: Cholesky factors of A_ii), `--reuse auto`, coarse problem on
rank 0 (n0 < 10000). GAMG = OpenFOAM-10 defaults, `faceAreaPair` agglomeration, DIC (`GAMG_DIC`) or
GaussSeidel (`GAMG_GS`) smoother. Times: every solve between two synchronizations of all processes;
Schwarz pays its initial setup, rebuilds and refactorizations, GAMG the construction of its coarse
matrices in every solve. Matrix assembly (OpenFOAM) and matrix generation (schwarz2lvl) are not
part of the costs.

## In short

| campaign | stopping test | 192³ | 256³ | 320³ |
|:--|:--|--:|--:|--:|
| `run_gamg_23038420`, 240 processes | ‖r‖₂/‖b‖₂ ≤ 1e-8 | **2.0×** | **4.3×** | – |
| scaling, 240 processes | Σ\|r\| ≤ 1e-2 Σ\|b\| | **1.8×** | **2.0×** | **2.0×** |
| scaling, 336 processes | Σ\|r\| ≤ 1e-2 Σ\|b\| | **2.2×** | **2.3×** | **1.9×** |
| scaling, 432 processes | Σ\|r\| ≤ 1e-2 Σ\|b\| | **2.7×** | **2.6×** | **2.4×** |
| BiCGStab, `--eig-tol 1e-2`, 240 processes (section 7) | Σ\|r\| ≤ 1e-2 Σ\|b\| | **2.0×** | **2.6×** | **2.5×** |
| PIMPLE-like loop, 240 processes (section 8) | relTol 0.05 / tolerance 1e-7, warm start | **0.96×** | **1.17×** | **1.18×** |
| PIMPLE-like loop, 432 processes (section 8) | relTol 0.05 / tolerance 1e-7, warm start | **1.81×** | **1.66×** | **1.35×** |

Cost per solve of GAMG_DIC (the fastest OpenFOAM variant) divided by the cost per solve of the
two-level Schwarz with AMG local solves, setup and updates included (> 1: Schwarz faster). Scaling
rows: GMRES(30), `--eig-tol 1e-4`; BiCGStab row: the setting to port (BiCGStab, `--nev 8 --eig-tol 1e-2
--eig-ncv 12`), cost with the machine stalls removed (≤ 2% below the measured one). PIMPLE-like rows:
the same setting, p_rgh source instead of α, every solve from the previous solution; cost per **time
step** (3 correctors, setup and rebuilds included, 24 steps), best OpenFOAM variant (GAMG_DIC) / Schwarz.

1. **Same problem, verified.** OpenFOAM assembles the generated matrix and right-hand side to
   3.6e-15 / 1.0e-14 (24³ case against `matgen`, gated: the jobs stop otherwise). 864 + 492 solves
   (scaling + `run_gamg_23038420`), all converged.
2. **GAMG needs more and more V-cycles as the mesh is refined; the Schwarz iterations grow
   mildly.** At 240 processes: GAMG 139 → 271 V-cycles per solve to 1e-8 (192³ → 256³) and
   32 → 58 → 69 to relTol 1e-2 (192³ → 256³ → 320³); Schwarz, right after a rebuild, 32 → 39 and
   11 → 14 → 15. The geometric agglomeration does not see the density jumps (see *Interpretation*).
3. **An iteration costs the same as a V-cycle** at 240 processes (13.4 / 14.6 ms at 192³,
   33.1 / 32.2 ms at 256³ to 1e-8; 34.8 / 34.5 and 68.9 / 68.8 ms at 256³ / 320³ to 1e-2): the whole
   gain is in the number of iterations, 2.0-2.9× fewer at 1e-2, 2.7-5.4× fewer at 1e-8.
4. **At the practical tolerance the gain is smaller but holds everywhere: 1.8-2.7×.** At 1e-2 the
   setup weighs more (27-35% of the Schwarz cost, against < 0.5% for GAMG), and the first two
   decades are the slowest ones of GMRES (5.5-8 iterations per decade after a rebuild, against 4-5
   over eight decades), while GAMG converges at its asymptotic rate from the first V-cycle (16 / 29
   V-cycles per decade at 1e-2 against 17 / 34 at 1e-8). At 256³ the ratio drops from 4.3× to 2.0×.
5. **Schwarz scales better.** Parallel efficiency 240 → 432 processes: Schwarz 0.85 / 0.96 / 1.18,
   GAMG 0.56 / 0.75 / 0.96 (192³ / 256³ / 320³). At 192³ GAMG does not speed up at all (0.567 →
   0.563 s per solve: its V-cycle gets slower, 17.4 → 19.4 ms). The ratio grows with the processes.
6. **Where the Schwarz time goes** (1e-2): setup and updates ~30%, local AMG V-cycles 21-33%,
   Krylov (SpMV, orthogonalization, stopping test) 17-27% — orthogonalization and stopping test
   alone 12-20% —, coarse correction 11-29% (growing with the processes: n0 = 8 per subdomain,
   solved on rank 0).
7. **The anomalies of the previous campaigns are gone**: initial setup / median rebuild = 0.98-1.03
   in all 9 runs (it was 11 and 15 in the first Schwarz run of the two GAMG jobs), regular rebuilds.
8. **BiCGStab instead of GMRES removes most of the Krylov overhead** (section 5): same preconditioner,
   1.04-1.20× the applications, each 0.72-0.85× as expensive; cost per solve 1.02 / 0.89 / 0.82× that
   of GMRES at 192³ / 256³ / 320³, so 1.8 / 2.2 / 2.3× cheaper than GAMG at 240 processes.
9. **A looser Arnoldi makes rebuilds cheaper at no cost in quality** (section 6): `--eig-tol 1e-2
   --eig-ncv 12` takes 27-29 operator applications instead of 42, rebuild −22-31%, initial setup
   −18%, same iterations after every rebuild. Reusing the factorization of Ã_ii between rebuilds
   saves only 14% of a rebuild on G100, and warm-starting the eigenvectors does not pay: neither
   was kept in the code.
10. **A larger coarse space does not pay** (section 7): `--nev 12 / 16` take 8-22% fewer
   applications, but each costs 6-28% more and the setup grows by 20-61%: 1.04-1.06× (nev 12) and
   1.15-1.20× (nev 16) the cost per solve of nev 8. The gain in iterations shrinks with the mesh
   (right after a rebuild at 320³: 16 / 15 / 15) and the aging of the coarse space is the same:
   `--nev 8` stays.
11. **In a PIMPLE-like loop the gain is smaller and lives in the last corrector** (section 8): p_rgh
   source of a VoF solver, warm start, correctors to relTol 0.05, last corrector to tolerance 1e-7.
   Schwarz is faster in 5 of 6 configurations (1.17-1.81× per time step, 1.24-1.90× without the
   first step) and equal at 192³ / 240 processes (0.96×, 1.01×). On the first two correctors GAMG is
   1.2-1.7× cheaper; on the last one Schwarz is 2.0-4.1× cheaper (19-38 applications against 34-72
   V-cycles on average). Setup and updates are 39-43% of the Schwarz cost. The one-level RAS costs
   2.9-5.7× GAMG. By the rule written before the run (≥ ~1.5× at both process counts) a port is not
   justified as it stands.

## 1. Reconnaissance: `run_gamg_23036378`

Five OpenFOAM variants at 192³ (20 steps × 3 correctors, 240 processes), to `relTol 1e-8` of the
OpenFOAM residual (true 2-norm residual up to 1.2e-7):

| variant | V-cycles / iterations per solve (avg / max) | cost per solve [s] |
|:--|--:|--:|
| GAMG, DIC smoother (`GAMG_DIC`) | 121.5 / 161 | **1.74** |
| GAMG, GaussSeidel (`GAMG_GS`, the usual p / p_rgh setup) | 186.2 / 285 | 2.57 |
| GAMG, `algebraicPair` agglomeration (`GAMG_ALG`) | 228.9 / 431 | 2.97 |
| PCG + 2 GAMG V-cycles (`PCG_GAMG`) | 59.0 / 112 | 4.75 |
| PCG + DIC (`PCG_DIC`, one level) | 2018.7 / 2059 | 2.40 |

- GAMG_DIC is the fastest OpenFOAM variant; the coefficient-based `algebraicPair` agglomeration is
  slower than the geometric one and much more erratic (max 431 at 192³, 935 at 256³); PCG with GAMG
  as preconditioner halves the iterations but each costs ~5.5 V-cycles of GAMG_DIC.
- **Not a valid comparison**: the matrix check failed (OpenFOAM matrix differed from the generated
  one by 0.5 relative to the largest entry, 2 of the 150 bubbles at another position). The bubble
  field compiled by the OpenFOAM toolchain of G100 (icpx oneAPI 2021, `-O3 -xCORE-AVX512`) drew
  different bubble positions; g++ and clang agree to 1e-15. Since then the bubble field of the
  application is compiled with the schwarz2lvl compiler and linked through a C interface, and the
  jobs stop if the matrices differ. The job hit the 70-minute limit in 256³ (PCG_GAMG, 8 of 48
  solves).

## 2. Asymptotic comparison: `run_gamg_23038420` (to 1e-8)

Same matrices (check `OK`, 3.6e-15), same accuracy: Schwarz to ‖r‖₂/‖b‖₂ = 1e-8, OpenFOAM to
`relTol 1e-9` of its residual (true 2-norm residual ≤ 1.1e-8). 240 processes; 192³ 20 steps, 256³
16 steps, 3 correctors.

| problem | variant | its avg / max | ms per iteration | cost per solve [s] | median [s] | vs GAMG_DIC |
|:--|:--|--:|--:|--:|--:|--:|
| 192³ | Schwarz `amg` | 50.7 / 131 | 13.4 | **1.03** | 0.55 | **0.51×** |
| | Schwarz `exact` | 35.9 / 52 | 28.3 | 1.23 | 1.10 | 0.60× |
| | GAMG_DIC | 139.1 / 182 | 14.6 | 2.03 | 2.00 | 1 |
| | GAMG_GS | 212.2 / 315 | 12.9 | 2.74 | 2.60 | 1.35× |
| | PCG_GAMG | 66.9 / 130 | 79.9 | 5.35 | 5.22 | 2.63× |
| 256³ | Schwarz `amg` | 50.6 / 93 | 33.1 | **2.04** | 1.51 | **0.23×** |
| | Schwarz `exact` | 42.9 / 62 | 84.1 | 4.52 | 3.82 | 0.52× |
| | GAMG_DIC | 270.8 / 357 | 32.2 | 8.72 | 8.32 | 1 |
| | GAMG_GS | 367.1 / 477 | 24.5 | 8.99 | 8.47 | 1.03× |

*median*: median cost of the solves after the first (no initial setup). Full tables:
[`report.md`](run_gamg_23038420/report.md).

- GAMG doubles its V-cycles from 192³ to 256³ (139 → 271, DIC; 212 → 367, GaussSeidel); the average
  Schwarz iterations stay at 51, right after a rebuild they grow from 32 to 39 (`exact`: 30 → 36).
- AMG local solves beat exact Cholesky factors here too (P7b): 2.1× / 2.5× cheaper per iteration
  for 1.4× / 1.2× more iterations.
- **Cold start of the first run (fixed afterwards, see 4.).** In the first Schwarz run of each job
  (`bub192_amg`, here and in `run_gamg_23036378`) the factorizations of Ã_ii in the eigensolver took
  14.3-15.0 s (18.2-18.6 s in the first job) on *every* process over the whole run, while a complete
  rebuild took 1.5 s; later runs of the same job did not show it (`bub192_exact`: initial setup
  2.0 s = rebuild 1.9 s). The initial setup (16.3 s, 21.6 s in the
  first job) entered the reuse policy as the cost of a rebuild, so the first rebuild came only after
  11 steps, at 131 iterations per solve (4.1× the fresh count); once a real rebuild had been measured
  (1.48 s) the policy rebuilt every 7 steps. At 256³ (no cold start) it rebuilt every 5-6 steps,
  at ≤ 93 iterations.

![Cost per solve, run_gamg_23038420](run_gamg_23038420/plots/gamg_cost.png)

## 3. Practical tolerance and scaling: `run_scaling_*` (relTol 1e-2)

**Stopping test.** In a time step OpenFOAM solves the pressure to `relTol` ~1e-2, never to 1e-8.
Both codes now use the OpenFOAM normalised residual with relTol 1e-2 (OpenFOAM: `relTol 1e-2;
tolerance 0;`; Schwarz: `--stop-norm openfoam --rtol 1e-2`, which evaluates Σ|r| of the GMRES
residual at every iteration). From x = 0 both stop at the first iteration with
Σ|b − Ax| ≤ 1e-2 Σ|b|: final normalised residual ≤ 9.96e-3 (Schwarz) and ≤ 9.98e-3 (GAMG); the
2-norm residual is then up to 6.3e-2 / 8.9e-2.

**Setup.** GAMG_DIC against Schwarz `amg`; 192³ / 256³ / 320³ (7.1M / 16.8M / 32.8M cells) on 240 /
336 / 432 processes (5 / 7 / 9 nodes, one job each), i.e. 16k to 137k cells per process; 16 steps ×
3 correctors for every size; scotch decomposition for OpenFOAM, METIS for schwarz2lvl. Binaries
built once and checked (`test_bubbles`, matrix check; sources `e12f07a96b18`); one unmeasured warm-up
run at the start of each job. 864 solves, all converged.

### Cost per solve [s]

| N \ processes | 240 | 336 | 432 |
|:--|--:|--:|--:|
| 192³ | S 0.316 · G 0.567 · **1.79×** | S 0.258 · G 0.570 · **2.21×** | S 0.206 · G 0.563 · **2.73×** |
| 256³ | S 0.986 · G 1.99 · **2.02×** | S 0.724 · G 1.65 · **2.28×** | S 0.573 · G 1.48 · **2.59×** |
| 320³ | S 2.38 · G 4.74 · **1.99×** | S 1.80 · G 3.39 · **1.88×** | S 1.12 · G 2.74 · **2.44×** |

S = Schwarz, G = GAMG_DIC, ratio G / S. Without the initial setup (cost of a long run if the rebuild
frequency stays the same) the ratio is 2.0-3.0×.

| N | processes | Schwarz iterations avg (min – max) | GAMG V-cycles avg (min – max) | ms per it. S / G | Schwarz setup share | rebuilds |
|--:|--:|--:|--:|--:|--:|--:|
| 192 | 240 | 15.3 (11 – 27) | 32.4 (25 – 43) | 14.4 / 17.4 | 30% | 1 |
| 192 | 336 | 15.1 (10 – 29) | 30.7 (21 – 43) | 11.5 / 18.5 | 32% | 2 |
| 192 | 432 | 13.1 (10 – 21) | 28.9 (21 – 38) | 10.6 / 19.4 | 33% | 2 |
| 256 | 240 | 20.6 (13 – 39) | 57.6 (48 – 90) | 34.8 / 34.5 | 27% | 1 |
| 256 | 336 | 18.4 (13 – 37) | 52.5 (43 – 69) | 26.0 / 31.4 | 34% | 2 |
| 256 | 432 | 17.7 (12 – 34) | 47.9 (41 – 61) | 21.6 / 30.9 | 33% | 2 |
| 320 | 240 | 24.6 (15 – 46) | 68.8 (54 – 112) | 68.9 / 68.8 | 29% | 1 |
| 320 | 336 | 23.4 (15 – 49) | 60.6 (49 – 82) | 50.3 / 55.9 | 35% | 2 |
| 320 | 432 | 20.3 (14 – 41) | 57.9 (48 – 96) | 39.5 / 47.2 | 28% | 1 |

*setup share*: (initial setup + rebuilds + refactorizations) / total cost. Rebuilds: after the
initial setup, every 6-8 steps; rebuild cost 0.8-13.9 s, refactorization 0.07-0.40 s.

### Phases [s per solve]

| N | processes | Schwarz: initial setup | updates | local solves | coarse corr. | Krylov | GAMG: setup | V-cycles |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 192 | 240 | 0.034 | 0.062 | 0.097 | 0.050 | 0.074 | 0.003 | 0.565 |
| 192 | 432 | 0.016 | 0.051 | 0.044 | 0.059 | 0.036 | 0.002 | 0.561 |
| 256 | 240 | 0.104 | 0.165 | 0.322 | 0.127 | 0.268 | 0.005 | 1.99 |
| 256 | 432 | 0.048 | 0.142 | 0.152 | 0.112 | 0.119 | 0.003 | 1.48 |
| 320 | 240 | 0.289 | 0.403 | 0.770 | 0.273 | 0.648 | 0.009 | 4.73 |
| 320 | 432 | 0.126 | 0.194 | 0.345 | 0.180 | 0.278 | 0.005 | 2.73 |

All 9 points: [`report.md`](scaling_report/report.md). The Schwarz iterations are split with the
timer averages of the logs. *Krylov* at 320³ / 240 processes: orthogonalization (CGS2) 0.36 s,
stopping test 0.11 s, SpMV 0.12 s per solve; orthogonalization and stopping test are 12-20% of the
Schwarz cost (the stopping test alone 3-7% of the GMRES time).

### Strong scaling: parallel efficiency vs 240 processes

| N | code | 336 | 432 |
|--:|:--|--:|--:|
| 192 | Schwarz | 0.88 | 0.85 |
| 192 | GAMG | 0.71 | 0.56 |
| 256 | Schwarz | 0.97 | 0.96 |
| 256 | GAMG | 0.86 | 0.75 |
| 320 | Schwarz | 0.94 | 1.18 |
| 320 | GAMG | 1.00 | 0.96 |

- **Schwarz**: smaller subdomains need fewer iterations (320³: 24.6 → 20.3) and a cheaper setup,
  which costs more than linearly in the subdomain size (initial setup 13.9 → 6.1 s for 1.8× fewer
  cells); the 1.18 at 320³ / 432 also contains one rebuild less than at 336 processes (the policy
  decides on measured times, so the number of rebuilds varies by one between runs). The coarse
  correction grows with the processes (n0 = 1920 → 3456, gathered on rank 0): 29% of the cost at
  192³ / 432, the limit at small subdomains (as in P7b at 960 processes).
- **GAMG**: at 16k-30k cells per process its V-cycle is latency-bound and gets slower with more
  processes (192³: 17.4 → 19.4 ms); its V-cycles decrease slightly with the processes (68.8 → 57.9
  at 320³; likely because the agglomeration is per process and the coarsest level grows with them).
- **Size**: from 192³ to 320³ (4.6× cells) the cost per solve grows 7.5× (Schwarz) / 8.4× (GAMG) at
  240 processes and 5.5× / 4.9× at 432.

![Strong scaling](scaling_report/plots/scaling_strong.png)

![Ratio of the costs](scaling_report/plots/scaling_ratio.png)

![Phases](scaling_report/plots/scaling_phases.png)

![Iterations of every solve](scaling_report/plots/scaling_iterations.png)

![Size scaling](scaling_report/plots/scaling_size.png)

## 4. The cold start, fixed

Since this campaign every run factorizes a 20³ Laplacian with the local backend before the time loop
(untimed), and every job starts with an unmeasured 96³ run. Result: initial setup / median rebuild
= 0.98-1.03 in all 9 runs, the first runs included; the warm-up took 0.66-1.16 s in the first run of
each job and 0.08-0.13 s in the others. The 14-18 s of the GAMG campaigns did not occur at all this
time, so their exact cause is not pinned down (one difference: those jobs compiled the code right
before the runs, these used binaries built by a separate job); the measured runs are protected
either way.

## 5. Krylov method: GMRES vs BiCGStab, `run_krylov_np240_r1e-2_23091304`

**Why.** In the scaling campaign the orthogonalization (CGS2) and the stopping test of GMRES took
23-28% of the time of the Schwarz iterations at 240 processes, growing with the mesh: their cost is
O(n_loc · j) per iteration, and both the rows per process and the iteration count j grow. BiCGStab
has a fixed amount of vector work per application of the preconditioner and explicit residuals, so
the OpenFOAM stopping test costs one sum in an existing reduction; it accepts the nonsymmetric
RAS/deflated preconditioner. CG was set aside: it needs a symmetric preconditioner, and ASM needs
~1.5× the iterations of RAS on these matrices, more than CG saves per iteration.

**Setup.** `--krylov gmres` (`amg`, the setting of section 3) against `--krylov bicgstab` (the
algorithm of OpenFOAM's `PBiCGStab`: right preconditioning, stopping test also at the half step),
same preconditioner (RAS + deflated, AMG local solves, `--tau 0.3 --nev 8 --eig-tol 1e-4`, reuse
`auto`), same matrices and stopping test as section 3 (OpenFOAM residual, relTol 1e-2), 16 steps ×
3 correctors, 240 processes, one job (5 nodes; sources `7b4fb7fcfae5`, `test_bicgstab` and
`test_stopping` passed on the compute node). *Iterations* = applications of the preconditioner for
both (BiCGStab: two per step). GAMG_DIC from `run_scaling_np240_23040226`. 288 solves, all converged,
final normalised residual ≤ 1.00e-2.

| N | Krylov | applications avg (min – max) | ms per application | Krylov [s/solve] | setup + updates [s/solve] | **cost per solve [s]** | BiCGStab / GMRES | GAMG / Schwarz |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| 192³ | GMRES(30) | 15.3 (11 – 27) | 13.6 | 0.063 | 0.097 | **0.306** | | 1.85× |
| | BiCGStab | 18.4 (11 – 39) | 11.6 | 0.048 | 0.096 | **0.311** | 1.02 | 1.83× |
| 256³ | GMRES(30) | 20.6 (13 – 39) | 34.9 | 0.271 | 0.274 | **0.995** | | 2.00× |
| | BiCGStab | 21.5 (15 – 41) | 28.4 | 0.158 | 0.279 | **0.888** | **0.89** | 2.24× |
| 320³ | GMRES(30) | 24.6 (15 – 46) | 72.0 | 0.721 | 0.714 | **2.48** | | 1.91× |
| | BiCGStab | 25.4 (16 – 44) | 51.9 | 0.289 | 0.709 | **2.03** | **0.82** | 2.34× |

*Krylov*: SpMV, orthogonalization / vector operations, stopping test and the rest of the Krylov
method, without the preconditioner. Without the initial setup the ratio is 1.02 / 0.88 / 0.79.
The GMRES runs reproduce section 3 to −3% / +1% / +4% (0.316 / 0.986 / 2.38 s), the run-to-run
spread of these measurements.

- **Where the gain comes from.** At 320³ the Krylov part drops from 0.72 to 0.29 s per solve:
  orthogonalization 0.41 → 0.11 s (CGS2 against ~12 vector operations per BiCGStab step), stopping
  test 0.12 → 0.015 s (no reconstruction of the residual from the basis). Local solves and coarse
  correction stay the same (1.05 → 1.03 s per solve) since BiCGStab needs only 4% more applications.
  The time per application is 0.85 / 0.81 / 0.72 of a GMRES iteration and keeps falling with the
  mesh, as the cost of CGS2 grows with n_loc · j.
- **Applications.** Per solve BiCGStab needs 1.17 / 1.06 / 1.05× the GMRES iterations (median),
  stopping at the half step in 21 / 30 / 15 of the 48 solves. At 192³ the ratio grows as the coarse
  space ages (27 vs 39 just before the rebuild), at 320³ it does not (46 vs 44): with one run per
  variant this is not a trend. A likely mechanism where it occurs: the bubbles that left the span of
  the old eigenvectors leave a few small isolated eigenvalues, which the minimal-residual GMRES
  removes one per iteration and the degree-1 ω-step of BiCGStab handles worse.
- **Same setup.** One rebuild in every run; at 256³ / 320³ BiCGStab rebuilds one step later
  (solve 28 vs 25, 25 vs 22): with cheaper iterations the aging coarse space wastes less time before
  the break-even with a rebuild. Setup and updates are now 31-35% of the Schwarz cost.
- **For the port.** BiCGStab here is the algorithm of OpenFOAM's `PBiCGStab`: the two-level Schwarz
  can be ported as an `lduMatrix::preconditioner` and used by the existing solver, which is
  faster than a GMRES of its own on these problems.
- Not measured: tighter tolerances (locally, 48³ on 8 processes, 1.15× the applications at relTol
  1e-8 against 1.09× at 1e-2) and other process counts.

![Cost per solve](run_krylov_np240_r1e-2_23091304/plots/cost_per_solve_np240.png)

![Time per application](run_krylov_np240_r1e-2_23091304/plots/time_per_iteration_np240.png)

![Every solve](run_krylov_np240_r1e-2_23091304/plots/solves_np240.png)

Plots: `python3 plot_krylov.py run_krylov_np240_r1e-2_23091304 --gamg run_scaling_np240_23040226`
(into the run directory); tables: `report_krylov.py` with the same arguments.

## 6. Rebuilds of the coarse space: `run_rebuild_np240_23093509`

**Why.** A rebuild at 320³ / 240 processes took ~14 s on the slowest rank (sections 3, 5), almost all
in the local eigensolver: factorization of Ã_ii + σI ~7 s, Arnoldi ~6.4 s (42 applications of the
operator, each a pair of triangular solves with those factors), kernel detection ~1.8 s; setup and
updates are 31-35% of the cost per solve. Three ideas were tried, two with a prototype that is not
kept in the code:

- **Reuse of the factorization of Ã_ii + σI** (prototype `--rebuild-reuse symbolic`): its pattern
  never changes in a time loop, so its symbolic analysis can be kept and only the numerical
  factorization redone; the kernel detection starts from the previous kernel. Same coarse space.
- **Warm-started eigenvectors** (block Krylov / Rayleigh-Ritz from the old eigenvectors, or Arnoldi
  started from a combination of them): tested locally only, as many operator applications as the
  cold Arnoldi (25-33 against 29). After one rebuild interval the eigenvectors near the threshold
  have moved (subspace distance 0.2 after one cell of rise, 0.96 after four), and what costs is the
  cluster of eigenvalues just above 1/τ (≈ 4.7-8 against 1/τ = 3.3), which a Krylov space must
  resolve whatever its starting vector: a block method of depth 5 is a polynomial of degree 5,
  Arnoldi with ncv = 28 one of degree 28.
- **Looser Arnoldi** (`--eig-tol 1e-2 --eig-ncv 12`, existing options): the same cluster explains why
  the coarse space barely changes. The bubble modes (large, well separated eigenvalues) converge in
  a few steps; the vectors of the cluster add little, and resolving each of them to 1e-4 is what
  costs. Locally: same iterations at relTol 1e-8 (25-27) with 13-23 applications instead of 42.

**Setup.** BiCGStab + RAS/deflated, AMG local solves, `--tau 0.3 --nev 8`, reuse `auto`; 256³ and
320³, 24 steps × 3 correctors, OpenFOAM test relTol 1e-2, 240 processes, one job (sources
`dd47448012f9`, built and checked on the compute node). Variants (cumulative): `base`
(`--eig-tol 1e-4`, the setting of section 5), `sym` (+ reuse of the factorization), `sym_eig`
(+ `--eig-tol 1e-2 --eig-ncv 12`).

**Machine disturbances.** `bub320_sym` timed out: its second solve took 375 s for 17 iterations (the
others ~0.9 s). Stalls of 5-10 s per solve also hit `base` at 320³ and `sym_eig` at 256³ (up to 14×
the median time per iteration), and in the next run (`bub320_sym_eig`) the warm-up of the direct
solver, which does not depend on any option, took 222 s instead of 0.08 s; the processes of the
killed run probably made it worse. The comparison therefore uses the rebuild times (from the CSV)
and the iterations, and a cost per solve with the stalls removed: setup and updates as measured,
the iterations at the median time per iteration of each run.

| N | variant | initial setup [s] | rebuilds [s] | vs base | applications (max) | iterations avg | after each rebuild | cost per solve [s], stalls removed | vs base | GAMG / variant |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 256³ | base | 5.00 | 5.15, 4.94 | | 42 | 21.8 | 16, 15, 15 | 0.832 | | 2.39 |
| | sym | 5.00 | 4.44, 4.25 | 0.86 | 42 | 20.9 | 16, 15, 14 | 0.789 | 0.95 | 2.52 |
| | sym_eig | 4.15 | 3.64, 3.48, 3.38 | 0.69 | 27 | 21.6 | 16, 15, 14, 15 | 0.819 | 0.98 | 2.43 |
| 320³ | base | 14.46 | 13.91, 13.52 | | 42 | 26.0 | 16, 16, 18 | 2.033 | | 2.33 |
| | sym | – | – | – | – | – | – | timed out | | |
| | sym_eig | 11.87 | 11.12, 10.31 | 0.78 | 29 | 25.2 | 16, 16, 19 | 1.872 | 0.92 | 2.53 |

*after each rebuild*: iterations of the first solve with a fresh coarse space (initial setup and every
rebuild), the quality of the coarse space. GAMG: cost per solve of `run_scaling_np240_23040226`
(16 steps). The `base` runs reproduce section 5 (0.888 / 2.03 s per solve with 16 steps).

- **The looser Arnoldi is the useful part.** 42 → 27-29 applications, Arnoldi −39% at 320³ (timer
  maxima per setup 6.0 → 3.7 s), rebuild −22% at 320³ and −31% at 256³ (together with the reuse),
  initial setup −17-18%; the iterations after every rebuild are the same (16 / 15 / 14-15 and
  16 / 16 / 18-19). At 256³ the cheaper rebuild makes the policy rebuild once more (3 instead of
  2), which keeps the cost per solve at −2%; at 320³ −8%.
- **The reuse of the factorization saves 14% of a rebuild** (256³: 5.04 → 4.35 s; cost per solve −5%),
  much less than locally (−37% of the factorization with 2 processes on 2 cores): with 48 processes per
  node the numerical factorization, bound by memory bandwidth, dominates, and the symbolic analysis
  it saves is ~20-25% of it. Not kept: one more set of factors per process in memory for ~5%.
- **What remains** of a rebuild is the numerical factorization of Ã_ii (~6 s at 320³). Reducing it
  would mean changing how the eigensolver solves with Ã_ii (e.g. iterative inner solves instead of
  the factorization): a research change with an uncertain gain, not pursued.

The run directory `run_rebuild_np240_23093509` (logs, CSV, plots of the prototype's report script)
and the prototype itself are not in the repository: the table above is the record of this campaign.
Sources `dd47448012f9` are the committed code plus the uncommitted prototype.

## 7. Size of the coarse space: `run_nev_np240_23094441`

**Why.** The cap `--nev 8` was chosen to 1e-8 with exact local solves (P5), when coarse solve and
eigensolver dominated the cost. At relTol 1e-2 with AMG local solves it binds in every subdomain
(m_i = 8 everywhere, τ = 0.3 never reached), and the iterations right after a rebuild still grow with
the mesh (11 → 14 → 15 in section 3, 16 → 16 → 18-19 in section 6). Question: does a larger coarse
space buy back mesh independence and fewer rebuilds, for a larger setup (Arnoldi) and a larger
coarse problem (n0 = nev × processes, on rank 0)?

**Setup.** `--nev 8 / 12 / 16` with `--eig-ncv` = 1.5 nev (12 / 18 / 24) and `--eig-tol 1e-2`
(section 6); BiCGStab + RAS/deflated, AMG local solves, τ = 0.3, reuse `auto`; 192³ / 256³ / 320³,
16 steps × 3 correctors, OpenFOAM test relTol 1e-2, 240 processes, one job (5 nodes; sources
`dd47448012f9`, built and checked on the compute node). GAMG_DIC from `run_scaling_np240_23040226`.
432 solves, all converged, final normalised residual ≤ 1.00e-2; no machine stalls (cost with stalls
removed ≤ 2% below the measured one).

| N | nev | n0 | Arnoldi applications (max) | after each rebuild | applications avg (min – max) | initial setup [s] | ms per application (coarse corr.) | **cost per solve [s]** | vs nev 8 | GAMG / Schwarz |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 192³ | 8 | 1920 | 29 | 12, 12 | 18.7 (11 – 37) | 1.31 | 10.5 (2.9) | **0.279** | | 2.03 |
| | 12 | 2880 | 40 | 10, 10 | 15.9 (10 – 30) | 1.63 | 11.9 (4.2) | **0.297** | 1.06 | 1.91 |
| | 16 | 3840 | 50 | 9, 10 | 14.5 (9 – 27) | 2.11 | 13.4 (5.8) | **0.335** | 1.20 | 1.69 |
| 256³ | 8 | 1920 | 29 | 16, 15 | 21.8 (14 – 41) | 4.21 | 24.6 (5.6) | **0.772** | | 2.58 |
| | 12 | 2880 | 40 | 14, 12 | 18.9 (12 – 37) | 5.24 | 26.5 (7.5) | **0.799** | 1.04 | 2.49 |
| | 16 | 3840 | 49 | 12, 11 | 17.7 (11 – 36) | 6.45 | 28.6 (9.6) | **0.887** | 1.15 | 2.25 |
| 320³ | 8 | 1920 | 35 | 16, 16 | 26.0 (16 – 46) | 12.1 | 49.0 (10.3) | **1.90** | | 2.49 |
| | 12 | 2880 | 40 | 15, 15 | 23.9 (15 – 45) | 14.6 | 51.9 (13.2) | **2.01** | 1.06 | 2.36 |
| | 16 | 3840 | 52 | 15, 15 | 23.9 (13 – 47) | 17.9 | 54.9 (16.2) | **2.26** | 1.19 | 2.10 |

m_i = nev in every subdomain for all variants. *after each rebuild*: applications of the first solve
with a fresh coarse space (initial setup and the one rebuild). *ms per application*: median of the
run; *coarse corr.*: coarse solves and the product with A of the deflated correction. *cost per
solve*: setup and updates included, stalls removed.

- **Fewer iterations, not less time in them.** nev 16 needs 22 / 19 / 8% fewer applications than
  nev 8 (192³ / 256³ / 320³), each 28 / 16 / 12% more expensive: the time of the iterations is the
  same (9.6 / 26.1 / 62.9 s per run with nev 8, 9.6 / 24.6 / 63.5 s with nev 16). The local solves
  per application do not change (6.1 / 15.3 / 31.1 ms); what grows is the part of the coarse
  correction that depends on nev, 2.0 → 5.0 ms per application at 192³ and 6.0 → 12.0 ms at 320³
  (timer maxima): restriction Zᵀr and prolongation Zy (products with the n_loc × nev local basis,
  linear in nev) and the coarse solve on rank 0, 1.1 → 1.9 → 3.0 ms whatever the mesh (n0 = 1920 /
  2880 / 3840; between linear and quadratic in nev, as the coarse matrix has an nev × nev block per
  pair of neighbouring subdomains). The smaller the subdomains, the larger its share: hence the 28% at
  192³.
- **The setup grows with nev and buys nothing.** Arnoldi: 35 / 40 / 52 operator applications at
  320³ with a basis of 12 / 18 / 24 vectors, 3.7 / 6.4 / 8.4 s per setup (timer maxima), while the
  factorization of Ã_ii + σI (7.4 s) and the kernel detection (1.7 s) do not depend on nev. Initial
  setup +20-25% (nev 12) and +48-61% (nev 16); every refactorization reassembles the coarse matrix
  Zᵀ A Z, 0.14 → 0.40 s per step at 320³ (nev 8 → 16). Of the +0.33 s per solve of nev 16 at 320³:
  initial setup and rebuild +0.24 s, refactorizations +0.09 s, iterations +0.01 s.
- **The gain shrinks with the mesh: no mesh independence bought back.** Right after a rebuild,
  12 → 16 → 16 applications with nev 8, 9.5 → 11.5 → 15 with nev 16 (192³ → 320³); at 320³ nev 12 and
  16 give the same count, one less than nev 8. Every subdomain has at least 16 eigenvalues below
  τ, so the coarse space is cut at λ_{nev+1}. The bubble modes are few (about 2 per subdomain: 150
  bubbles of diameter 0.08 in 240 subdomains of side ~0.16) and well separated (section 6): nev 8
  already contains them, and the extra vectors come from the cluster just below τ. Why that cluster
  gets denser with the mesh, as an estimate (the eigenvalues are not printed by the G100 runs): with
  an overlap of one cell, the small eigenvalues that are not bubble modes belong to fields smooth in
  the subdomain whose energy lies mostly outside the overlap; in the GenEO setting their number below
  a fixed level grows with the interface measured in cells, ~(H/h)², i.e. 2.7× from 192³ to 320³ at
  240 processes (H/h ≈ 31 → 51). A fixed number of extra vectors then raises λ_{nev+1} less and less;
  a mesh-independent coarse space needs m_i growing like (H/h)², the cost the cap is there to avoid.
- **Aging and rebuilds unchanged.** One rebuild in every run, at the same step (nev 16 at 320³ one
  step later: its rebuild costs 45% more, so the break-even comes later). Before the rebuild the
  applications reach 2.6-3.1× the fresh count for all three sizes of the coarse space: the extra
  eigenvectors are modes of the old matrix as well, and a bubble that has risen 3-4 cells needs a mode
  constant on its new position, which more modes of the old matrix do not represent better. Fewer
  rebuilds would need a coarse space that follows the bubbles, not a larger one.
- **nev 8 stays.** nev 12 costs 4-6% more, nev 16 15-20%; neither wins on any mesh. The run also
  confirms section 6 on 16-step runs: nev 8 with `--eig-tol 1e-2 --eig-ncv 12` costs 0.283 / 0.780 /
  1.94 s per solve against 0.311 / 0.888 / 2.03 s with `--eig-tol 1e-4` in section 5 (−9 / −12 / −4%,
  different jobs): 2.0 / 2.6 / 2.5× cheaper than GAMG at 240 processes (stalls removed), the
  configuration to port.
- **432 processes not run.** With smaller subdomains (16k-76k cells, H/h ≈ 25-42) the iteration gain of
  nev 16 is expected at most as at 192³-256³ here (−19-22%), while the coarse problem doubles to
  n0 = 6912 on rank 0 and the local solves per application are ~1.8× cheaper: the coarse correction,
  already 29% of the cost at 192³ / 432 with nev 8 (section 3), weighs more, and the extra cost per
  application is expected above the 12-28% measured at 240 processes. The second job (9 nodes, ~30 min) was
  cancelled.

The run directory `run_nev_np240_23094441` (report, CSV, logs, plots) and the scripts of this
campaign (`Allrun_nev.sh`, `submit_nev.sh`, `report_nev.py`) are not in the repository: the tables of
this section are the record of the campaign. Sources `dd47448012f9` (section 6); the variants do not
use the prototype of section 6.

## 8. Warm start, p_rgh source and PIMPLE tolerances: `run_warm_np240_23097350`, `run_warm_np432_23097354`

**Why.** Sections 1-7 solve every system from 0 with b = vol·α. That right-hand side is constant on
every bubble, i.e. close to the near-kernel modes of D⁻¹A (one per bubble, nearly constant on it) that
the coarse space captures and the `faceAreaPair` agglomeration does not: locally (48³, 25 bubbles,
8 subdomains) 99% of the energy of the initial error lay in the 10 lowest modes of (A, diag A). An
OpenFOAM time loop differs in three ways, each of which lowers that weight: the source of p_rgh lives
on the interfaces (1% in the same modes, from 0), every solve starts from the previous solution
(< 0.1% with both), and only the last corrector of a step goes to a tight tolerance. This campaign
repeats the comparison in those conditions, the last check before a port. Decision rule written before
the run (`README.md`): port if the two-level Schwarz is ≥ ~1.5× faster than the best OpenFOAM variant
at both process counts.

**Setup.**

- *Systems*: the matrices of sections 3-7 (bubbles3d, 150 bubbles, ρ ratio 1000, 0.5 cells of rise
  per step); right-hand side the p_rgh source of interFoam for the fluid at rest,
  A p = −div(φ_g), φ_g = −gh_f snGrad(ρ)_f β_f |S_f|, gh_f = 1 − z_f (`--bubble-source buoyancy`,
  bubblesDict `source buoyancy;`): fluxes across the interfaces only, zero sum over every bubble.
- *Time loop*: 24 steps × 3 correctors, every solve from the previous solution (`--warm-start 1`,
  `bubblyPressureFoam -warmStart`). The first two correctors stop at relTol 0.05 or at tolerance 1e-7
  of the OpenFOAM normalised residual, the last corrector of every step at tolerance 1e-7 only
  (relTol 0; `W_GAMGFinal`, `--final-rtol 0 --final-atol 1e-7`): the p_rgh / p_rghFinal controls of
  the interFoam tutorials. The reuse policy keeps one reference q per corrector (main README, *Setup
  reuse*): right after a rebuild q differs by a factor 1.2-2.4 between the correctors (highest in
  the last one), which a single reference would count as aging.
- *Variants*: `s2l_2L`, the setting to port of section 7 (BiCGStab + RAS/deflated, AMG local solves,
  `--tau 0.3 --nev 8 --eig-tol 1e-2 --eig-ncv 12`, reuse `auto`); `s2l_1L`, BiCGStab + one-level RAS
  with the same AMG local solves (no eigensolver, no rebuilds); `W_GAMG`, GAMG with the DIC smoother
  (`faceAreaPair`, OpenFOAM-10 defaults) for every corrector; `W_GAMG_PCG`, the same for the first two
  correctors and PCG + 2 GAMG V-cycles for the last one (192³ only).
- *Grid*: 192³ / 256³ / 320³ on 240 and 432 processes (5 and 9 nodes, one job each), the OpenFOAM
  cases of section 3 (scotch; METIS for schwarz2lvl).
- *Checks*: sources `729b64f0bbbf` (commit `0edbd85` plus the files of this campaign; the jobs ran
  from an extracted copy, so `log.git` reads "not a git checkout"), built on the compute node;
  OpenFOAM matrices and right-hand sides equal to the generator for both sources (24³: 3.6e-15 /
  2.5e-15), parallel assembly of the right-hand side equal to the serial one (4 processes, 8.1e-15 of
  Σ|b|); `test_warmstart`, `test_stopping`, `test_bicgstab` passed. Both codes solve the same
  sequence: the initial normalised residual of the first corrector of every step agrees to ≤ 3.1e-5
  (relative) between schwarz2lvl and GAMG. 1440 solves, all converged except one of `s2l_1L`
  (320³ / 432 processes, last corrector of step 0: 1000 applications, 7.9e-7).

### Cost per time step [s]

| N | processes | cells / process | Schwarz 2-level | RAS 1-level | GAMG_DIC | GAMG, PCG_GAMG last | **GAMG / Schwarz 2-level** | section 3 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 192³ | 240 | 29k | 0.605 | 3.27 | 0.581 | 1.23 | **0.96×** (1.01×) | 1.79× |
| 256³ | 240 | 70k | 1.68 | 11.2 | 1.97 | – | **1.17×** (1.24×) | 2.02× |
| 320³ | 240 | 137k | 4.30 | 26.2 | 5.08 | – | **1.18×** (1.27×) | 1.99× |
| 192³ | 432 | 16k | 0.358 | 1.90 | 0.647 | 1.58 | **1.81×** (1.90×) | 2.73× |
| 256³ | 432 | 39k | 0.911 | 6.85 | 1.51 | – | **1.66×** (1.75×) | 2.59× |
| 320³ | 432 | 76k | 2.06 | 15.2 | 2.79 | – | **1.35×** (1.46×) | 2.44× |

The three correctors of a step, setup and updates included, whole run / 24 steps; in parentheses
steps 1-23 only (no initial setup, no solve from 0): what a long run pays per step. GAMG_DIC is the best
OpenFOAM variant everywhere. *section 3*: GAMG_DIC / two-level Schwarz, cost per solve, b = vol·α,
every solve from 0, relTol 1e-2, GMRES. RAS 1-level at 320³ / 432: one solve not converged.

### Where the time goes [s per time step]

| N | processes | Schwarz: correctors 1-2 | Schwarz: last corrector | Schwarz: setup + updates | GAMG: correctors 1-2 | GAMG: last corrector | last corrector, GAMG / Schwarz |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 192³ | 240 | 0.096 | 0.259 | 0.250 (41%) | 0.066 | 0.508 | 1.96× |
| 256³ | 240 | 0.247 | 0.742 | 0.691 (41%) | 0.190 | 1.763 | 2.37× |
| 320³ | 240 | 0.539 | 1.929 | 1.836 (43%) | 0.443 | 4.608 | 2.39× |
| 192³ | 432 | 0.073 | 0.145 | 0.141 (39%) | 0.044 | 0.599 | 4.14× |
| 256³ | 432 | 0.162 | 0.381 | 0.368 (40%) | 0.111 | 1.391 | 3.65× |
| 320³ | 432 | 0.320 | 0.905 | 0.840 (41%) | 0.240 | 2.535 | 2.80× |

*setup + updates*: initial setup, rebuilds and refactorizations (share of the Schwarz cost). The setup
of GAMG (coarse matrices) is < 1% of its cost and not shown.

### Iterations per solve

| N | processes | Schwarz 2-level: corrector 1 / 2 / last (min – max) | GAMG_DIC: corrector 1 / 2 / last (min – max) | ms per it. Schwarz / GAMG | RAS 1-level: last |
|--:|--:|--:|--:|--:|--:|
| 192³ | 240 | 3.0 / 5.4 / 22.5 (12 – 41) | 2.0 / 4.1 / 35.2 (25 – 45) | 11.5 / 13.9 | 419 |
| 256³ | 240 | 3.0 / 6.2 / 29.8 (15 – 66) | 2.0 / 4.5 / 56.2 (42 – 85) | 25.4 / 31.1 | 579 |
| 320³ | 240 | 3.0 / 7.1 / 38.4 (18 – 65) | 2.0 / 5.1 / 71.9 (51 – 120) | 50.8 / 63.9 | 672 |
| 192³ | 432 | 3.2 / 4.7 / 19.3 (10 – 39) | 2.0 / 4.1 / 33.8 (23 – 43) | 7.97 / 16.1 | 454 |
| 256³ | 432 | 3.2 / 6.2 / 24.0 (12 – 49) | 2.0 / 5.1 / 53.0 (35 – 98) | 16.3 / 25.0 | 637 |
| 320³ | 432 | 3.2 / 6.9 / 30.5 (13 – 64) | 2.0 / 5.2 / 59.7 (42 – 97) | 30.2 / 41.5 | 700 |

Averages over the 24 steps. Schwarz: applications of the preconditioner (two per BiCGStab step);
GAMG: V-cycles. The last corrector starts at a normalised residual of ~2-8e-4: 3.3-3.7 decades to 1e-7.

- **The gain is in the last corrector.** There the two-level Schwarz is 2.0-2.4× (240 processes) and
  2.8-4.1× (432) cheaper than GAMG: 5.4-10.3 applications per decade against 10.2-21.0 V-cycles, each
  application 0.50-0.83× as expensive as a V-cycle. On the first two correctors (relTol 0.05, 1.3
  decades) GAMG is 1.2-1.7× cheaper: 2 + 4-5 V-cycles against 3 + 5-7 applications. GAMG spends 88-93%
  of its time in the last corrector, the two-level Schwarz 41-45%.
- **Setup and rebuilds take the rest.** Setup and updates are 39-43% of the Schwarz cost (27-35% in
  sections 3-5): the iterations got cheaper, the setup did not. The eigensolver alone is 16-26% of the
  cost (rebuilds of 0.66-11.8 s, two per run). Built up phase by phase, the ratio best OpenFOAM /
  Schwarz is 1.64 / 1.99 / 2.06 (240) and 2.98 / 2.78 / 2.28 (432) on the iterations only, 1.31 /
  1.68 / 1.80 and 2.34 / 2.24 / 1.94 with the refactorizations, 0.96 / 1.17 / 1.18 and 1.81 / 1.66 /
  1.35 with setup and rebuilds (192³ / 256³ / 320³).
- **Aging.** Right after a rebuild the last corrector needs 11-23 applications; in 8-12 steps (4-6 cells
  of rise) it grows 2.9-4.0×, up to or slightly above the V-cycles of GAMG, and the break-even rule
  rebuilds there (two rebuilds in every run, at steps 8-11 and 18-23). GAMG shows no trend, its
  agglomeration does not depend on the bubbles, but varies 1.8-2.8× (min to max) with their
  configuration.
- **Scaling.** From 240 to 432 processes the parallel efficiency is 0.94 / 1.02 / 1.16 for Schwarz and
  0.50 / 0.72 / 1.01 for GAMG (192³ / 256³ / 320³): at 192³ GAMG is slower on 432 processes than on
  240 (0.581 → 0.647 s per step, V-cycle 13.9 → 16.1 ms), as in section 3. Hence the ratio grows with
  the processes.
- **One level is not enough.** The one-level RAS keeps up on the first two correctors (4-5 + 8-11
  applications) but needs 420-700 applications in the last one (120-210 per decade): 2.9-5.7× the cost
  of GAMG. With p = 0 on the top face only, the lowest pressure modes span the whole column and a
  one-level method moves information by one subdomain per iteration: the coarse space is what makes
  the tight final tolerance affordable.
- **PCG + GAMG for the last corrector** needs 13 iterations instead of 34 V-cycles, but each costs
  62-81 ms (two V-cycles with the DICGaussSeidel smoother): 2.1-2.4× the cost per step of GAMG_DIC at
  192³.
- **Decision.** By the rule written before the run a port is not justified as it stands: ≥ 1.5× only at
  432 processes for 192³ and 256³ (16k-39k cells per process); at 240 processes (29k-137k cells)
  0.96-1.18×, a margin that the conversion LDU → CSR, the decomposition of OpenFOAM and its processor
  interfaces can absorb. The two-level Schwarz is still faster than GAMG in 5 of 6 configurations; the
  part of its cost left to reduce is keeping the coarse space up to date (next steps).

![Ratio built up phase by phase](warm_plots/ratio.png)

![Cost per time step by corrector](warm_plots/correctors.png)

![Cost per time step by phase](warm_plots/phases.png)

![Iterations per solve](warm_plots/iterations.png)

![Last corrector along the time loop](warm_plots/last_corrector.png)

![Time per iteration](warm_plots/time_per_iteration.png)

Full tables: [`warm_report/report.md`](warm_report/report.md); numbers of the plots:
[`warm_plots/phases.csv`](warm_plots/phases.csv). Reproduce them from `test/gamg`:

```bash
python3 report_warm.py run_warm_np240_23097350 run_warm_np432_23097354 \
    --ref run_scaling_np240_23040226 --ref run_scaling_np432_23040228     # -> warm_report/
python3 plot_warm.py run_warm_np240_23097350 run_warm_np432_23097354        # -> warm_plots/
```

Not measured: the two effects separately (`SOURCE=alpha` and `WARM=0` jobs of `Allrun_warm.sh`), GAMG
with the GaussSeidel smoother.

## Interpretation

- **Why GAMG needs more V-cycles on finer meshes.** With β_gas = 1000 β_liquid, a pressure field that
  is constant inside a bubble and changes only in the liquid around it has an energy ∫β|∇p|²
  ~1000 times smaller than a field of the same shape with its gradient inside the gas: there is one
  such near-kernel mode per bubble. A coarse level can remove it only if its aggregates follow the
  bubbles. The `faceAreaPair` agglomeration pairs cells by face area: on a uniform mesh the
  aggregates ignore the interfaces and straddle them, so these modes are left to the smoother. DIC
  or Gauss-Seidel reduce them by local propagation, a few cells per sweep, while a bubble spans
  more cells as the mesh is refined (diameter 15 cells at 192³, 26 at 320³): more V-cycles. The
  coefficient-based `algebraicPair` does not fix this in OpenFOAM (run 1: slower and erratic).
- **Why the Schwarz iterations grow only mildly.** The coarse space contains, in every subdomain,
  the eigenvectors of smallest eigenvalue of the local generalized eigenproblem: precisely the modes
  that are nearly constant on each bubble (piece) in the subdomain, whatever their shape. What
  remains grows slowly with the subdomain size (after a rebuild: 11 → 15 iterations at 1e-2,
  32 → 39 at 1e-8, the same with exact local solves: 30 → 36, so not the AMG approximation). The
  coarse space is capped by `--nev 8` in every subdomain (m_i = 8 everywhere, τ = 0.3 never
  reached). Raising the cap to 16 does not remove the growth (section 7): it lowers the count at
  192³ / 256³ but not at 320³, consistent with a cluster of local eigenvalues below τ that gets
  denser as the subdomains grow in cells; a fixed cap only shifts the growth.
- **The price: the coarse space follows the bubbles.** The iterations grow 2-3× within 6-8 steps
  (3-4 cells of rise) and the basis must be rebuilt (eigensolver: 0.8-13.9 s); with the
  refactorizations this is ~30% of the Schwarz cost at 1e-2. GAMG pays nothing for the motion (setup
  < 0.5%), because its agglomeration does not depend on the coefficients — the same reason why it
  converges slowly.
- **Why the gain moves into the last corrector (section 8).** With the p_rgh source (fluxes across
  the interfaces, zero net source over every bubble) and a warm start, little of the initial error lies
  in the bubble modes (locally < 0.1% of its energy in the 10 lowest modes, against 99% for b = vol·α
  from 0). The 1.3 decades of the first two correctors are then removed by GAMG in 2 + 4-5 V-cycles
  and the coarse space has nothing to add, while each application still pays for it. The last
  corrector reduces the residual by 3.3-3.7 decades: there the error left is in the bubble modes,
  GAMG needs 10-21 V-cycles per decade and the two-level Schwarz 5-10 applications. Even there GAMG
  converges faster than in sections 2-3 (17 / 34 V-cycles per decade to 1e-8 from 0 at 192³ / 256³):
  b = vol·α overstated the difficulty of the problem for GAMG.
- **For the port.** The gain is algorithmic (iterations), with iterations costing as much as
  V-cycles, so it should survive a port to OpenFOAM, provided the subdomain structure and the
  coarse space live across time steps (e.g. in a `MeshObject`) and the conversion of the LDU
  matrix stays cheap. Section 8 qualifies it: in a PIMPLE loop the gain is 0.96-1.81× per time
  step, concentrated in the last corrector, and the setup and rebuilds of the coarse space (~40% of
  the Schwarz cost) are the part a port would not reduce.

## Caveats

- Sections 1-7: every solve starts from 0 with b = vol·α. Section 8 measures the warm start with
  the p_rgh source of the fluid at rest: the source of a real flow also contains div(HbyA), which
  carries the velocity field and is not confined to the interfaces; each code starts from its own
  previous solution (the two agree, see the checks of section 8).
- Section 8: one run per configuration (run-to-run spread of the earlier campaigns ±3-4%); GAMG with
  the GaussSeidel smoother, the usual p_rgh setting, not run (1.03-1.35× slower than DIC to 1e-8 in
  section 2).
- One problem (bubbles, density ratio 1000), OpenFOAM 10 while the target is OpenFOAM 14 (GAMG is
  essentially unchanged), different decompositions (scotch / METIS), no LDU → CSR conversion.
- The matrix generation of schwarz2lvl runs on rank 0 and is not timed: 37-156 s per run, more than
  the timed cost, so the wall time of a Schwarz run (50-271 s) is larger than that of GAMG (35-254 s)
  although its solves are 1.8-2.7× cheaper. A property of the test harness, not of the
  preconditioner (OpenFOAM assembles in parallel), but more than half of the Schwarz core-hours.

## Next steps suggested by these numbers

1. ~~PCG instead of GMRES~~ → **BiCGStab** (section 5): 0.82-1.02× the cost of GMRES with the same
   preconditioner. The Krylov overhead (vector operations, stopping test, rest; SpMV excluded) is
   now ~13% of the iterations at 320³; the cost is in the preconditioner (local solves ~58%, coarse
   correction ~20% of the iterations) and in setup and updates (31-35% of the cost).
2. ~~Cheaper rebuilds~~ (section 6): `--eig-tol 1e-2 --eig-ncv 12` (rebuild −22-31%, initial setup
   −18%, same coarse space) is the setting to use; the reuse of the factorization of Ã_ii (−14% per
   rebuild) and warm-started eigenvectors (no gain) are not kept. Selective rebuilds (only the
   subdomains whose matrix changed) would save core-hours but not wall time here: the slowest rank
   sets the time and 70-85% of the subdomains contain a bubble.
3. ~~Coarse space size~~ (section 7): `--nev 12 / 16` cut the applications by 8-22% but cost
   4-20% more per solve (coarse correction per application and setup grow with nev, the gain
   vanishes at 320³, same aging); `--nev 8` stays. The setting to port: BiCGStab, `--tau 0.3
   --nev 8 --eig-tol 1e-2 --eig-ncv 12`, AMG local solves, reuse `auto`.
4. ~~Warm start in both codes~~ (section 8): with the p_rgh source and PIMPLE tolerances the
   two-level Schwarz is 0.96-1.81× cheaper per time step than GAMG; the gain is in the last
   corrector, setup and rebuilds are ~40% of its cost. By the rule written before the run a port is
   not justified as it stands.
5. **Harness**: distributed generation of the matrix values (more than halves the Schwarz test cost).
6. **A coarse space that follows the bubbles** (section 8): the aging (2.9-4.0× more applications in
   the last corrector within 8-12 steps) and the rebuilds (eigensolver 16-26% of the cost) are the part
   of the cost left to reduce. The modes that age are those nearly constant on bubbles that have
   moved: e.g. the GenEO vectors of the last rebuild plus, at every step, indicator vectors of the gas
   regions of every subdomain computed from the current coefficients in O(n_i) and assembled in A_00
   at every refactorization. Not tested: a research step, not a refinement.
