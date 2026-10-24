# Scaling campaign: GAMG vs two-level Schwarz

Runs: `run_scaling_np240_23040226`, `run_scaling_np336_23040227`, `run_scaling_np432_23040228`  
Problem: bubbles3d, 150 bubbles, density ratio 1000, CFL 0.5; 16 steps × 3 correctors; every solve from 0.  
Stopping test (both codes): OpenFOAM normalised residual, relTol **1e-2** (Σ|b − Ax| ≤ relTol Σ|b|).  
Schwarz: `--tau 0.3 --nev 8 --eig-tol 1e-4 --local-approx amg --coarse-solver cpardiso --coarse-procs 48 --coarse-root-below 10000 --stop-norm openfoam`; OpenFOAM: `GAMG_DIC` (OpenFOAM-10).

## Checks

- `run_scaling_np240_23040226` binaries: sources=e12f07a96b18 test_bubbles=passed matrix_check="OK: matrices equal to 3.6e-15, right-hand sides to 1.0e-14 (relative to the largest entry; tolerance 1e-12)" 2026-10-23 23:38:56
- `run_scaling_np336_23040227` binaries: sources=e12f07a96b18 test_bubbles=passed matrix_check="OK: matrices equal to 3.6e-15, right-hand sides to 1.0e-14 (relative to the largest entry; tolerance 1e-12)" 2026-10-23 23:38:56
- `run_scaling_np432_23040228` binaries: sources=e12f07a96b18 test_bubbles=passed matrix_check="OK: matrices equal to 3.6e-15, right-hand sides to 1.0e-14 (relative to the largest entry; tolerance 1e-12)" 2026-10-23 23:38:56
- solves: 864, converged 864; final normalised residual (max): Schwarz 9.96e-03, GAMG 9.98e-03; true 2-norm residual ||b − Ax||₂/||b||₂ (max): Schwarz 6.31e-02, GAMG 8.85e-02
- runs not used (not ok or incomplete): none
- warm-up of the direct solver in the first (unmeasured) run of each job: run_scaling_np240_23040226 0.66 s, run_scaling_np336_23040227 1.16 s, run_scaling_np432_23040228 0.67 s
- warm-up in the measured runs (max): 0.13 s; initial setup / median rebuild: 192³/240 1.01, 192³/336 1.01, 192³/432 1.03, 256³/240 1.00, 256³/336 1.01, 256³/432 0.98, 320³/240 1.02, 320³/336 1.02, 320³/432 1.03 (≈ 1 expected; 11 in the first schwarz2lvl run of run_gamg_23038420: cold first factorization)

## Cost per solve [s] (setup amortized over the solves of the run)

| N \ processes | 240 | 336 | 432 |
|:--|--:|--:|--:|
| 192³ (7.1M) | S 0.316 · G 0.567 · **1.79×** | S 0.258 · G 0.570 · **2.21×** | S 0.206 · G 0.563 · **2.73×** |
| 256³ (16.8M) | S 0.986 · G 1.99 · **2.02×** | S 0.724 · G 1.65 · **2.28×** | S 0.573 · G 1.48 · **2.59×** |
| 320³ (32.8M) | S 2.38 · G 4.74 · **1.99×** | S 1.80 · G 3.39 · **1.88×** | S 1.12 · G 2.74 · **2.44×** |

S = two-level Schwarz (AMG local solves), G = GAMG (GAMG_DIC); **ratio** = G / S (> 1: Schwarz faster).

## Details

| N | NP | code | its avg (min-max) | time/it [ms] | initial setup [s] | updates [s] | iterations [s] | cost/solve [s] | steady/solve [s] | wall [s] |
|--:|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 192 | 240 | Schwarz | 15.3 (11-27) | 14.4 | 1.61 | 2.99 (1 reb., 14 ref.) | 10.6 | 0.316 | 0.283 | 57 |
| 192 | 240 | GAMG | 32.4 (25-43) | 17.4 | 0.042 | 0.073 | 27.1 | 0.567 | 0.566 | 35 |
| 192 | 336 | Schwarz | 15.1 (10-29) | 11.5 | 1.00 | 2.98 (2 reb., 13 ref.) | 8.38 | 0.258 | 0.237 | 51 |
| 192 | 336 | GAMG | 30.7 (21-43) | 18.5 | 0.025 | 0.051 | 27.3 | 0.570 | 0.569 | 35 |
| 192 | 432 | Schwarz | 13.1 (10-21) | 10.6 | 0.791 | 2.44 (2 reb., 13 ref.) | 6.66 | 0.206 | 0.189 | 50 |
| 192 | 432 | GAMG | 28.9 (21-38) | 19.4 | 0.032 | 0.043 | 26.9 | 0.563 | 0.562 | 36 |
| 256 | 240 | Schwarz | 20.6 (13-39) | 34.8 | 4.99 | 7.94 (1 reb., 14 ref.) | 34.4 | 0.986 | 0.882 | 132 |
| 256 | 240 | GAMG | 57.6 (48-90) | 34.5 | 0.057 | 0.195 | 95.4 | 1.99 | 1.99 | 103 |
| 256 | 336 | Schwarz | 18.4 (13-37) | 26.0 | 3.19 | 8.54 (2 reb., 13 ref.) | 23.0 | 0.724 | 0.657 | 119 |
| 256 | 336 | GAMG | 52.5 (43-69) | 31.4 | 0.045 | 0.126 | 79.0 | 1.65 | 1.65 | 84 |
| 256 | 432 | Schwarz | 17.7 (12-34) | 21.6 | 2.32 | 6.84 (2 reb., 13 ref.) | 18.3 | 0.573 | 0.525 | 114 |
| 256 | 432 | GAMG | 47.9 (41-61) | 30.9 | 0.038 | 0.098 | 71.1 | 1.48 | 1.48 | 76 |
| 320 | 240 | Schwarz | 24.6 (15-46) | 68.9 | 13.9 | 19.3 (1 reb., 14 ref.) | 81.2 | 2.38 | 2.09 | 271 |
| 320 | 240 | GAMG | 68.8 (54-112) | 68.8 | 0.104 | 0.360 | 227 | 4.74 | 4.74 | 254 |
| 320 | 336 | Schwarz | 23.4 (15-49) | 50.3 | 8.78 | 21.1 (2 reb., 13 ref.) | 56.6 | 1.80 | 1.62 | 244 |
| 320 | 336 | GAMG | 60.6 (49-82) | 55.9 | 0.071 | 0.243 | 163 | 3.39 | 3.39 | 168 |
| 320 | 432 | Schwarz | 20.3 (14-41) | 39.5 | 6.05 | 9.32 (1 reb., 14 ref.) | 38.6 | 1.12 | 0.997 | 213 |
| 320 | 432 | GAMG | 57.9 (48-96) | 47.2 | 0.060 | 0.197 | 131 | 2.74 | 2.74 | 137 |

*steady/solve* = (updates + iterations) / solves, without the initial setup (the cost per solve of a long simulation, if the rebuild frequency stays the same). *wall* = whole run, with start-up, mesh reading / matrix generation and assembly (not compared).

## Phases [s per solve]

| N | NP | Schwarz: initial setup | updates | local solves | coarse corr. | Krylov | GAMG: initial setup | updates | V-cycles |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 192 | 240 | 0.034 | 0.062 | 0.097 | 0.050 | 0.074 | 0.001 | 0.002 | 0.565 |
| 192 | 336 | 0.021 | 0.062 | 0.066 | 0.059 | 0.050 | 0.001 | 0.001 | 0.568 |
| 192 | 432 | 0.016 | 0.051 | 0.044 | 0.059 | 0.036 | 0.001 | 0.001 | 0.561 |
| 256 | 240 | 0.104 | 0.165 | 0.322 | 0.127 | 0.268 | 0.001 | 0.004 | 1.99 |
| 256 | 336 | 0.067 | 0.178 | 0.204 | 0.115 | 0.159 | 0.001 | 0.003 | 1.65 |
| 256 | 432 | 0.048 | 0.142 | 0.152 | 0.112 | 0.119 | 0.001 | 0.002 | 1.48 |
| 320 | 240 | 0.289 | 0.403 | 0.770 | 0.273 | 0.648 | 0.002 | 0.007 | 4.73 |
| 320 | 336 | 0.183 | 0.440 | 0.518 | 0.224 | 0.439 | 0.001 | 0.005 | 3.39 |
| 320 | 432 | 0.126 | 0.194 | 0.345 | 0.180 | 0.278 | 0.001 | 0.004 | 2.73 |

Schwarz iterations split with the timer averages of `log.bub<N>_amg` (Krylov = SpMV, orthogonalization, stopping test and vector updates of GMRES).

## Strong scaling: parallel efficiency vs 240 processes

| N | code | 240 | 336 | 432 |
|--:|:--|--:|--:|--:|
| 192 | Schwarz | 1.00 | 0.88 | 0.85 |
| 192 | GAMG | 1.00 | 0.71 | 0.56 |
| 256 | Schwarz | 1.00 | 0.97 | 0.96 |
| 256 | GAMG | 1.00 | 0.86 | 0.75 |
| 320 | Schwarz | 1.00 | 0.94 | 1.18 |
| 320 | GAMG | 1.00 | 1.00 | 0.96 |

![scaling_strong](plots/scaling_strong.png)

![scaling_size](plots/scaling_size.png)

![scaling_ratio](plots/scaling_ratio.png)

![scaling_phases](plots/scaling_phases.png)

![scaling_iterations](plots/scaling_iterations.png)

