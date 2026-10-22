# P5 - size of the coarse space and coarse solver: run_p5_23013515

`NP=240 NODES=5 COARSE='--coarse-solver cpardiso --coarse-procs 48 --coarse-root-below 10000' STEPS=30 STEPS_CS=4 CORR=3 N_SEQ=128 N_LARGE=192 CS_PROCS='5 20 48'`

`not a git checkout: see src_hash.txt sources md5: b41f63a702b7`

Runs not ok: none

All solves: 948, converged 948, max true residual 1.00e-08

## Sweep: coarse vectors per subdomain (--reuse auto)

m_i avg = mean n0 over the solves / processes; m_i max = last coarse space. *fresh its* = first solve after every (re)build. **vs ref** compares the median-time cost.

| run | tau | nev | m_i avg / max | n0 | coarse solver | its avg / max | fresh its | refact. / rebuilds | mean refact. [s] | mean rebuild [s] | cost/solve [s] | cost/solve, median [s] | vs ref |
|:--|--:|--:|--:|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| bub128_ref | 0.3 | 60 | 25.1 / 35 | 6025 | root | 20.2 / 39 | 12.2 | 26 / 3 | 0.233 | 2.727 | 0.469 | 0.465 | ref |
| bub128_t010 | 0.1 | 60 | 5.4 / 8 | 1288 | root | 33.7 / 61 | 22.3 | 27 / 2 | 0.045 | 2.383 | 0.309 | 0.305 | -35% |
| bub128_n16 | 0.3 | 16 | 15.9 / 16 | 3824 | root | 18.7 / 30 | 15.0 | 25 / 4 | 0.094 | 0.677 | 0.234 | 0.232 | -50% |
| bub128_n8 | 0.3 | 8 | 8.0 / 8 | 1920 | root | 23.2 / 35 | 19.2 | 25 / 4 | 0.051 | 0.473 | 0.199 | 0.195 | -58% |
| bub192_ref | 0.3 | 60 | 52.5 / 60 | 12594 | cpardiso 48 (mtype 2) | 23.7 / 44 | 13.0 | 27 / 2 | 1.302 | 13.458 | 2.104 | 2.089 | ref |
| bub192_t010 | 0.1 | 60 | 9.6 / 15 | 2295 | root | 35.2 / 64 | 21.2 | 26 / 3 | 0.321 | 10.229 | 1.585 | 1.560 | -25% |
| bub192_t003 | 0.03 | 60 | 2.7 / 5 | 646 | root | 61.4 / 107 | 44.5 | 26 / 3 | 0.270 | 10.115 | 2.197 | 2.172 | +4% |
| bub192_n32 | 0.3 | 32 | 31.9 / 32 | 7668 | root | 20.9 / 36 | 15.0 | 26 / 3 | 0.763 | 6.138 | 1.349 | 1.336 | -36% |
| bub192_n16 | 0.3 | 16 | 16.0 / 16 | 3840 | root | 23.3 / 42 | 18.4 | 25 / 4 | 0.377 | 3.080 | 1.011 | 0.995 | -52% |
| bub192_n8 | 0.3 | 8 | 8.0 / 8 | 1920 | root | 28.2 / 42 | 24.2 | 25 / 4 | 0.292 | 2.165 | 1.008 | 0.993 | -52% |

### Where an iteration and an update go

Per GMRES iteration [ms], timers averaged over the ranks: solves with the factors of A_ii, coarse solve (A_00^{-1}, waiting included), products with V (restrict + prolong), the rest (SpMV, orthogonalization). Per operation [s]: eigensolver and A_00 factorization per setup, A_00 recomputation and refactorization per refactorize step.

| run | local [ms] | coarse [ms] | V products [ms] | rest [ms] | total [ms] | eigensolver [s] | A_00 fact. setup [s] | A_00 = V^T A V refact. [s] | A_00 refact. [s] |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| bub128_ref | 3.79 | 7.49 | 0.65 | 1.94 | 13.87 | 2.00 | 0.356 | 0.027 | 0.170 |
| bub128_t010 | 3.72 | 0.72 | 0.10 | 1.86 | 6.41 | 2.01 | 0.019 | 0.002 | 0.006 |
| bub128_n16 | 3.77 | 3.15 | 0.38 | 1.78 | 9.07 | 0.40 | 0.106 | 0.014 | 0.043 |
| bub128_n8 | 3.74 | 1.08 | 0.16 | 1.80 | 6.78 | 0.29 | 0.026 | 0.005 | 0.009 |
| bub192_ref | 19.69 | 18.66 | 5.74 | 7.14 | 51.24 | 9.19 | 2.372 | 0.429 | 0.614 |
| bub192_t010 | 19.54 | 1.72 | 1.12 | 7.04 | 29.41 | 9.09 | 0.074 | 0.026 | 0.033 |
| bub192_t003 | 19.57 | 0.56 | 0.27 | 6.78 | 27.18 | 9.09 | 0.020 | 0.004 | 0.006 |
| bub192_n32 | 19.72 | 11.04 | 3.62 | 6.48 | 40.86 | 4.00 | 0.540 | 0.231 | 0.276 |
| bub192_n16 | 19.64 | 3.28 | 1.98 | 6.58 | 31.48 | 2.08 | 0.111 | 0.074 | 0.045 |
| bub192_n8 | 19.61 | 1.20 | 0.98 | 6.74 | 28.53 | 1.43 | 0.029 | 0.023 | 0.010 |

![sweep](plots/p5_sweep.png)

Iterations of the first corrector of every step (`R` = rebuild of the coarse basis):

- `bub128_ref`: 13 12 14 17 19 24 28 32 36 12R 13 14 17 19 21 26 33 39 12R 13 13 16 18 22 26 29 31 12R 12 13
- `bub128_t010`: 22 22 24 26 29 34 37 43 46 50 54 22R 22 23 26 29 36 46 51 59 59 23R 23 23 23 25 27 31 35 40
- `bub128_n16`: 15 15 16 18 21 25 30 15R 15 15 16 18 21 27 15R 15 15 18 22 25 15R 15 15 16 18 21 24 28 15R 15
- `bub128_n8`: 20 20 21 23 26 30 35 19R 19 20 21 23 27 32 19R 20 21 23 27 32 19R 19 19 21 22 24 28 32 19R 19
- `bub192_ref`: 13 13 14 16 19 23 26 30 34 42 42 13R 13 15 18 20 23 27 30 34 41 13R 13 14 16 19 24 30 34 41
- `bub192_t010`: 21 21 23 26 29 41 49 57 63 21R 21 21 24 26 31 37 45 54 59 21R 21 22 24 29 36 42 50 56 59 22R
- `bub192_t003`: 45 45 51 57 63 71 79 86 46R 47 48 49 54 68 78 97 97 44R 44 45 56 72 86 96 43R 43 45 49 56 72
- `bub192_n32`: 15 15 16 17 20 25 29 33 15R 15 16 18 20 24 28 34 15R 15 15 18 21 25 29 36 15R 15 15 18 22 27
- `bub192_n16`: 18 18 19 20 25 30 41 18R 18 19 22 25 29 33 19R 18 19 20 24 28 33 18R 18 19 20 25 29 37 19R 20
- `bub192_n8`: 25 25 26 27 32 41 25R 25 25 25 28 33 38 24R 24 24 27 30 37 24R 23 24 27 32 40 23R 24 26 29 34

## Coarse solver at the ref setting (--reuse refactor)

Same matrices and coarse space in every run: only where A_00 is factorized and solved changes. cpardiso K = K leaders, each owning the rows of N/K consecutive ranks (K = nodes: one per node).

| run | coarse solver | n0 | A_00 fact. setup [s] | A_00 refact. [s] | coarse solve [ms/it] | iteration [ms] | cost/solve, median [s] |
|:--|:--|--:|--:|--:|--:|--:|--:|
| bub192_cs_root | root | 12589 | 2.026 | 1.163 | 28.80 | 61.15 | 2.423 |
| bub192_cs_cp5 | cpardiso 5 (mtype 2) | 12589 | 2.420 | 0.844 | 19.78 | 52.02 | 2.249 |
| bub192_cs_cp20 | cpardiso 20 (mtype 2) | 12589 | 2.216 | 0.647 | 15.08 | 47.79 | 2.118 |
| bub192_cs_cp48 | cpardiso 48 (mtype 2) | 12589 | 2.186 | 0.557 | 15.66 | 47.22 | 2.102 |

