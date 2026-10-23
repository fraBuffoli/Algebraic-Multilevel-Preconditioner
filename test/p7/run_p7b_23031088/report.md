# P7 - inexact local solvers: run_p7b_23031088

`NP=240 COARSE='--coarse-solver cpardiso --coarse-procs 48 --coarse-root-below 10000' BASE='--tau 0.3 --nev 8 --eig-tol 1e-4' STEPS_A=20 STEPS_B=16 CORR=3 N_A=192 N_B=256`

`not a git checkout: see src_hash.txt sources md5: 882d3a3ae513`

test_inexact: ALL PASSED

test_reuse: ALL PASSED

Runs not ok: none

All solves: 588, converged 588, max true residual 9.97e-09

## Cost per solve (--reuse auto)

*A_ii solver* as reported by rank 0 (AMG: levels and operator complexity = nonzeros of all levels / nnz(A_ii)). *fresh its* = first solve after every (re)build. *refact.* = mean time of a refactorize step (A_ii: numerical refactorization or AMG rebuild; A_00 recomputed), *rebuild* = mean time of a full setup (eigensolver included). **vs ref** / **vs sp** compare the median-time cost with exact factors in double / single precision.

| run | A_ii solver | AMG levels / op. compl. | its avg / max | fresh its | refact. / rebuilds | refact. [s] | rebuild [s] | cost/solve [s] | cost/solve, median [s] | vs ref | vs sp |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| bub192_ref | pardiso cholesky | - | 28.2 / 42 | 24.5 | 16 / 3 | 0.308 | 1.933 | 1.017 | 0.999 | ref | +28% |
| bub192_sa | amg sa+gs V(1,1), coarsest <= 500 | 3 / 1.56 | 29.9 / 50 | 24.7 | 17 / 2 | 0.095 | 1.572 | 0.517 | 0.509 | -49% | -35% |
| bub192_sp | pardiso cholesky single | - | 32.0 / 51 | 27.3 | 17 / 2 | 0.236 | 1.944 | 0.794 | 0.777 | -22% | sp |
| bub192_rs | amg rs+gs V(1,1), coarsest <= 500 | 5 / 2.81 | 29.8 / 51 | 24.7 | 17 / 2 | 0.119 | 1.574 | 0.645 | 0.634 | -37% | -18% |
| bub192_sasp | amg sa+gs V(1,1), coarsest <= 500 single | 3 / 1.56 | 32.2 / 50 | 27.7 | 17 / 2 | 0.092 | 1.569 | 0.495 | 0.487 | -51% | -37% |
| bub256_ref | pardiso cholesky | - | 32.7 / 50 | 27.0 | 12 / 3 | 1.437 | 6.678 | 3.691 | 3.678 | ref | +35% |
| bub256_sa | amg sa+gs V(1,1), coarsest <= 500 | 4 / 1.58 | 38.4 / 73 | 30.7 | 13 / 2 | 0.200 | 4.983 | 1.640 | 1.626 | -56% | -40% |
| bub256_sp | pardiso cholesky single | - | 38.6 / 73 | 30.0 | 13 / 2 | 0.955 | 6.163 | 2.728 | 2.721 | -26% | sp |
| bub256_rs | amg rs+gs V(1,1), coarsest <= 500 | 5 / 2.85 | 36.9 / 71 | 27.3 | 13 / 2 | 0.294 | 5.043 | 1.969 | 1.950 | -47% | -28% |
| bub256_sasp | amg sa+gs V(1,1), coarsest <= 500 single | 4 / 1.58 | 43.1 / 97 | 32.7 | 13 / 2 | 0.192 | 4.949 | 1.626 | 1.601 | -56% | -41% |
| bub256_sare | amg sa+gs V(1,1), coarsest <= 500, reused hierarchy | 4 / 1.58 | 41.6 / 76 | 30.7 | 13 / 2 | 0.162 | 4.985 | 1.719 | 1.724 | -53% | -37% |

## Where an iteration and a setup go

Per GMRES iteration [ms], timers averaged over the ranks: local solves (A_ii factors or approximate inverse), coarse solve (A_00^{-1}, waiting included), products with V (restrict + prolong), the rest (SpMV, orthogonalization, halo). Per operation [s]: A_ii factorization / AMG setup per (re)build, A_ii refactorization / AMG rebuild per refactorize step, eigensolver per (re)build (A~_ii is always factorized exactly). *solves* = share of the median-time cost spent in GMRES.

| run | local [ms] | coarse [ms] | V products [ms] | rest [ms] | total [ms] | A_ii setup [s] | A_ii refact. [s] | eigensolver [s] | solves |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| bub192_ref | 19.61 | 1.20 | 0.98 | 6.68 | 28.48 | 0.467 | 0.181 | 1.18 | 79% |
| bub192_sa | 5.75 | 1.19 | 0.98 | 5.79 | 13.72 | 0.058 | 0.047 | 1.19 | 79% |
| bub192_sp | 11.09 | 1.25 | 1.00 | 6.15 | 19.48 | 0.523 | 0.141 | 1.18 | 78% |
| bub192_rs | 9.47 | 1.23 | 0.98 | 6.11 | 17.79 | 0.083 | 0.070 | 1.19 | 82% |
| bub192_sasp | 4.22 | 1.28 | 1.00 | 5.61 | 12.10 | 0.053 | 0.042 | 1.19 | 78% |
| bub256_ref | 63.30 | 1.28 | 2.52 | 17.80 | 84.90 | 1.829 | 0.938 | 3.92 | 75% |
| bub256_sa | 14.43 | 1.22 | 2.53 | 15.02 | 33.18 | 0.117 | 0.107 | 3.94 | 77% |
| bub256_sp | 34.47 | 1.26 | 2.55 | 15.62 | 53.90 | 1.308 | 0.567 | 3.93 | 76% |
| bub256_rs | 23.60 | 1.24 | 2.51 | 15.23 | 42.58 | 0.206 | 0.172 | 3.94 | 80% |
| bub256_sasp | 10.56 | 1.24 | 2.54 | 14.95 | 29.29 | 0.107 | 0.098 | 3.94 | 77% |
| bub256_sare | 14.41 | 1.24 | 2.52 | 14.55 | 32.72 | 0.123 | 0.069 | 3.94 | 79% |

![cost and iterations](plots/p7_cost.png)

![where an iteration goes](plots/p7_iteration.png)

Iterations of the first corrector of every step (`R` = rebuild of the coarse basis):

- `bub192_ref` (exact): 25 25 26 27 32 41 25R 25 25 25 28 33 38 24R 24 24 27 30 37 24R
- `bub192_sa` (amg sa+gs): 25 25 26 27 31 38 49 25R 25 25 27 30 34 38 48 24R 24 24 25 29
- `bub192_sp` (exact, single): 28 28 29 30 33 42 51 27R 28 27 29 32 34 39 48 27R 26 26 28 31
- `bub192_rs` (amg rs+gs): 25 25 26 28 31 42 51 25R 25 25 26 30 33 39 24R 24 25 27 31 38
- `bub192_sasp` (amg sa+gs, single): 28 28 29 30 33 38 49 28R 28 28 30 33 35 38 48 27R 27 27 28 32
- `bub256_ref` (exact): 27 28 30 36 50 27R 29 32 45 27R 27 29 32 35 44 27R
- `bub256_sa` (amg sa+gs): 30 29 31 36 50 64 33R 30 37 49 64 29R 29 32 33 37
- `bub256_sp` (exact, single): 30 31 33 37 50 65 30R 30 35 51 63 30R 30 32 34 36
- `bub256_rs` (amg rs+gs): 27 28 30 36 50 65 28R 29 34 47 63 27R 28 30 32 36
- `bub256_sasp` (amg sa+gs, single): 33 32 33 36 50 64 34R 33 37 49 64 90 31R 33 33 36
- `bub256_sare` (amg sa+gs, reuse): 30 32 37 46 56 32R 33 40 55 71 30R 31 33 39 45 52

