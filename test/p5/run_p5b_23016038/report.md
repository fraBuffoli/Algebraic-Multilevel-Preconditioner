# P5 - size of the coarse space and coarse solver: run_p5b_23016038

`NP=960 COARSE='--coarse-solver cpardiso --coarse-procs 48 --coarse-root-below 10000' STEPS_A=30 STEPS_B=24 CORR=3 N_A=192 N_B=256`

`not a git checkout: see src_hash.txt sources md5: b41f63a702b7`

Runs not ok: none

All solves: 1296, converged 1296, max true residual 1.00e-08

## Sweep: coarse vectors per subdomain (--reuse auto)

m_i avg = mean n0 over the solves / processes; m_i max = last coarse space. *fresh its* = first solve after every (re)build. **vs ref** compares the median-time cost.

| run | options | m_i avg / max | n0 | coarse solver | its avg / max | fresh its | refact. / rebuilds | mean refact. [s] | mean rebuild [s] | cost/solve [s] | cost/solve, median [s] | vs ref |
|:--|:--|--:|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| bub192_ref | tau 0.3, nev 60 | 24.5 / 38 | 23480 | cpardiso 48 (mtype 2) | 18.8 / 39 | 13.7 | 24 / 5 | 0.857 | 3.800 | 1.335 | 1.327 | ref |
| bub192_n16 | tau 0.3, nev 16 | 15.9 / 16 | 15304 | cpardiso 48 (mtype 2) | 17.5 / 26 | 15.0 | 23 / 6 | 0.346 | 1.135 | 0.827 | 0.822 | -38% |
| bub192_n16sp | tau 0.3, nev 16, single | 15.9 / 16 | 15303 | cpardiso 48 (mtype 2) | 19.2 / 29 | 17.0 | 23 / 6 | 0.338 | 1.159 | 1.019 | 1.029 | -22% |
| bub192_n16e4 | tau 0.3, nev 16, tol 1e-4 | 15.9 / 16 | 15303 | cpardiso 48 (mtype 2) | 16.8 / 24 | 15.0 | 22 / 7 | 0.365 | 1.102 | 0.893 | 0.888 | -33% |
| bub192_n16spe4 | tau 0.3, nev 16, single, tol 1e-4 | 15.9 / 16 | 15304 | cpardiso 48 (mtype 2) | 19.5 / 28 | 17.0 | 23 / 6 | 0.330 | 1.127 | 0.965 | 0.969 | -27% |
| bub192_n8 | tau 0.3, nev 8 | 8.0 / 8 | 7680 | root | 22.4 / 40 | 18.7 | 24 / 5 | 0.127 | 0.548 | 0.369 | 0.356 | -73% |
| bub192_n8sp | tau 0.3, nev 8, single | 8.0 / 8 | 7680 | root | 23.9 / 35 | 20.8 | 24 / 5 | 0.126 | 0.538 | 0.370 | 0.347 | -74% |
| bub192_n8spe4 | tau 0.3, nev 8, single, tol 1e-4 | 8.0 / 8 | 7680 | root | 23.8 / 34 | 20.8 | 24 / 5 | 0.128 | 0.519 | 0.361 | 0.345 | -74% |
| bub256_ref | tau 0.3, nev 60 | 41.8 / 58 | 40080 | cpardiso 48 (mtype 2) | 20.5 / 45 | 13.0 | 20 / 3 | 3.654 | 13.424 | 3.331 | 3.056 | ref |
| bub256_n16 | tau 0.3, nev 16 | 16.0 / 16 | 15360 | cpardiso 48 (mtype 2) | 21.5 / 39 | 18.0 | 19 / 4 | 0.454 | 2.191 | 1.374 | 1.383 | -55% |
| bub256_n16sp | tau 0.3, nev 16, single | 16.0 / 16 | 15360 | cpardiso 48 (mtype 2) | 24.5 / 39 | 20.0 | 19 / 4 | 0.545 | 2.739 | 1.433 | 1.368 | -55% |
| bub256_n16e4 | tau 0.3, nev 16, tol 1e-4 | 16.0 / 16 | 15360 | cpardiso 48 (mtype 2) | 21.5 / 39 | 18.0 | 19 / 4 | 0.446 | 1.987 | 1.278 | 1.273 | -58% |
| bub256_n16spe4 | tau 0.3, nev 16, single, tol 1e-4 | 16.0 / 16 | 15360 | cpardiso 48 (mtype 2) | 23.4 / 39 | 20.0 | 19 / 4 | 0.439 | 1.940 | 1.302 | 1.305 | -57% |
| bub256_n8 | tau 0.3, nev 8 | 8.0 / 8 | 7680 | root | 24.7 / 39 | 21.2 | 19 / 4 | 0.222 | 1.297 | 0.729 | 0.695 | -77% |
| bub256_n8sp | tau 0.3, nev 8, single | 8.0 / 8 | 7680 | root | 27.2 / 44 | 23.2 | 19 / 4 | 0.208 | 1.271 | 0.669 | 0.629 | -79% |
| bub256_n8spe4 | tau 0.3, nev 8, single, tol 1e-4 | 8.0 / 8 | 7680 | root | 27.2 / 44 | 23.2 | 19 / 4 | 0.207 | 1.155 | 0.654 | 0.624 | -80% |

### Where an iteration and an update go

Per GMRES iteration [ms], timers averaged over the ranks: solves with the factors of A_ii, coarse solve (A_00^{-1}, waiting included), products with V (restrict + prolong), the rest (SpMV, orthogonalization). Per operation [s]: eigensolver and A_00 factorization per setup, A_00 recomputation and refactorization per refactorize step.

| run | local [ms] | coarse [ms] | V products [ms] | rest [ms] | total [ms] | eigensolver [s] | A_00 fact. setup [s] | A_00 = V^T A V refact. [s] | A_00 refact. [s] |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| bub192_ref | 3.12 | 37.04 | 0.52 | 3.61 | 44.30 | 1.70 | 1.693 | 0.022 | 0.794 |
| bub192_n16 | 3.12 | 30.60 | 0.32 | 3.02 | 37.05 | 0.34 | 0.637 | 0.012 | 0.283 |
| bub192_n16sp | 1.87 | 38.00 | 0.33 | 3.31 | 43.52 | 0.34 | 0.664 | 0.012 | 0.292 |
| bub192_n16e4 | 3.12 | 34.76 | 0.32 | 3.63 | 41.83 | 0.29 | 0.653 | 0.012 | 0.314 |
| bub192_n16spe4 | 1.83 | 35.12 | 0.32 | 3.29 | 40.57 | 0.29 | 0.667 | 0.012 | 0.285 |
| bub192_n8 | 2.99 | 6.99 | 0.14 | 3.06 | 13.18 | 0.25 | 0.184 | 0.004 | 0.088 |
| bub192_n8sp | 1.72 | 7.31 | 0.14 | 3.31 | 12.48 | 0.25 | 0.174 | 0.004 | 0.088 |
| bub192_n8spe4 | 1.72 | 7.24 | 0.14 | 3.11 | 12.21 | 0.20 | 0.179 | 0.004 | 0.089 |
| bub256_ref | 10.00 | 57.15 | 2.75 | 7.47 | 77.37 | 5.03 | 6.726 | 0.180 | 3.338 |
| bub256_n16 | 9.98 | 33.88 | 1.11 | 6.16 | 51.13 | 1.06 | 0.640 | 0.038 | 0.289 |
| bub256_n16sp | 5.79 | 32.60 | 1.14 | 5.59 | 45.12 | 1.06 | 0.660 | 0.038 | 0.341 |
| bub256_n16e4 | 9.98 | 29.90 | 1.10 | 6.02 | 47.00 | 0.90 | 0.660 | 0.039 | 0.284 |
| bub256_n16spe4 | 5.80 | 31.85 | 1.14 | 5.78 | 44.57 | 0.90 | 0.584 | 0.039 | 0.289 |
| bub256_n8 | 9.90 | 7.01 | 0.47 | 6.03 | 23.42 | 0.76 | 0.175 | 0.011 | 0.088 |
| bub256_n8sp | 5.69 | 7.14 | 0.48 | 5.88 | 19.18 | 0.76 | 0.171 | 0.011 | 0.086 |
| bub256_n8spe4 | 5.69 | 7.20 | 0.48 | 5.61 | 18.97 | 0.60 | 0.173 | 0.011 | 0.085 |

![sweep](plots/p5_sweep.png)

Iterations of the first corrector of every step (`R` = rebuild of the coarse basis):

- `bub192_ref`: 14 14 15 19 24 30 38 13R 14 15 17 20 25 14R 14 17 21 25 14R 14 16 22 27 13R 13 15 18 22 26 14R
- `bub192_n16`: 15 15 16 20 26 15R 15 16 19 24 15R 15 16 20 26 15R 15 17 21 15R 15 17 22 15R 15 16 19 23 15R 15
- `bub192_n16sp`: 17 17 18 22 29 17R 17 18 21 26 17R 17 18 22 17R 17 19 24 17R 17 19 24 17R 17 18 21 25 17R 17 17
- `bub192_n16e4`: 15 15 16 20 15R 15 15 19 24 15R 15 16 20 15R 15 17 22 15R 15 17 23 15R 15 16 19 15R 15 16 19 15R
- `bub192_n16spe4`: 17 17 18 22 28 17R 17 18 21 26 17R 17 18 22 28 17R 17 19 23 17R 17 19 24 17R 17 18 21 25 17R 17
- `bub192_n8`: 19 20 21 24 29 37 19R 19 19 22 25 30 19R 19 19 23 29 18R 18 19 24 31 19R 20 20 23 27 18R 19 20
- `bub192_n8sp`: 21 22 23 26 32 20R 21 21 23 27 21R 21 21 24 28 35 21R 21 21 27 35 21R 21 22 25 30 21R 21 22 23
- `bub192_n8spe4`: 21 22 23 26 32 20R 21 21 23 27 21R 21 21 24 29 21R 21 21 26 33 21R 21 24 27 21R 21 21 22 27 33
- `bub256_ref`: 13 13 16 23 31 42 13R 13 15 20 25 33 13R 13 15 19 26 33 13R 13 16 20 24 27
- `bub256_n16`: 18 18 19 26 39 18R 18 18 23 31 18R 18 19 21 28 18R 18 19 25 30 18R 18 19 22
- `bub256_n16sp`: 20 20 21 28 39 20R 20 20 25 33 20R 20 21 23 30 35 20R 20 21 26 32 34 20R 20
- `bub256_n16e4`: 18 18 19 26 39 18R 18 18 23 31 18R 18 19 21 28 18R 18 19 25 30 18R 18 19 22
- `bub256_n16spe4`: 20 20 21 28 39 20R 20 20 25 33 20R 20 21 23 30 20R 20 21 27 32 20R 20 21 24
- `bub256_n8`: 21 21 22 28 39 21R 21 22 25 32 21R 22 22 25 33 21R 21 22 27 34 22R 21 22 24
- `bub256_n8sp`: 23 23 24 30 40 23R 23 24 27 33 43 23R 23 24 28 35 23R 23 24 29 35 24R 24 24
- `bub256_n8spe4`: 23 23 24 30 40 23R 23 24 27 33 43 23R 23 24 28 35 23R 23 24 29 34 24R 24 24

