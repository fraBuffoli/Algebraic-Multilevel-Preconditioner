# P7 - inexact local solvers: run_p7b_23031087

`NP=960 COARSE='--coarse-solver cpardiso --coarse-procs 48 --coarse-root-below 10000' BASE='--tau 0.3 --nev 8 --eig-tol 1e-4' STEPS_A=20 STEPS_B=16 CORR=3 N_A=192 N_B=256`

`not a git checkout: see src_hash.txt sources md5: 882d3a3ae513`

Runs not ok: none

All solves: 264, converged 264, max true residual 9.92e-09

## Cost per solve (--reuse auto)

*A_ii solver* as reported by rank 0 (AMG: levels and operator complexity = nonzeros of all levels / nnz(A_ii)). *fresh its* = first solve after every (re)build. *refact.* = mean time of a refactorize step (A_ii: numerical refactorization or AMG rebuild; A_00 recomputed), *rebuild* = mean time of a full setup (eigensolver included). **vs ref** / **vs sp** compare the median-time cost with exact factors in double / single precision.

| run | A_ii solver | AMG levels / op. compl. | its avg / max | fresh its | refact. / rebuilds | refact. [s] | rebuild [s] | cost/solve [s] | cost/solve, median [s] | vs ref | vs sp |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| bub192_ref | pardiso cholesky | - | 23.6 / 52 | 18.8 | 16 / 3 | 0.128 | 0.499 | 0.403 | 0.376 | ref | - |
| bub192_sa | amg sa+gs V(1,1), coarsest <= 500 | 3 / 1.51 | 22.1 / 32 | 19.2 | 16 / 3 | 0.106 | 0.459 | 0.290 | 0.277 | -26% | - |
| bub256_ref | pardiso cholesky | - | 24.9 / 39 | 21.0 | 12 / 3 | 0.224 | 1.159 | 0.701 | 0.675 | ref | +14% |
| bub256_sa | amg sa+gs V(1,1), coarsest <= 500 | 3 / 1.53 | 26.5 / 43 | 22.0 | 13 / 2 | 0.133 | 1.040 | 0.524 | 0.475 | -30% | -20% |
| bub256_sp | pardiso cholesky single | - | 27.9 / 44 | 23.0 | 13 / 2 | 0.208 | 1.138 | 0.617 | 0.594 | -12% | sp |

## Where an iteration and a setup go

Per GMRES iteration [ms], timers averaged over the ranks: local solves (A_ii factors or approximate inverse), coarse solve (A_00^{-1}, waiting included), products with V (restrict + prolong), the rest (SpMV, orthogonalization, halo). Per operation [s]: A_ii factorization / AMG setup per (re)build, A_ii refactorization / AMG rebuild per refactorize step, eigensolver per (re)build (A~_ii is always factorized exactly). *solves* = share of the median-time cost spent in GMRES.

| run | local [ms] | coarse [ms] | V products [ms] | rest [ms] | total [ms] | A_ii setup [s] | A_ii refact. [s] | eigensolver [s] | solves |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| bub192_ref | 2.96 | 6.69 | 0.13 | 3.04 | 12.82 | 0.472 | 0.022 | 0.26 | 74% |
| bub192_sa | 0.99 | 6.86 | 0.14 | 2.39 | 10.38 | 0.012 | 0.010 | 0.22 | 79% |
| bub256_ref | 9.74 | 6.63 | 0.46 | 5.03 | 21.85 | 0.270 | 0.077 | 0.60 | 77% |
| bub256_sa | 3.21 | 6.83 | 0.47 | 5.35 | 15.86 | 0.041 | 0.026 | 0.62 | 79% |
| bub256_sp | 5.61 | 6.73 | 0.47 | 4.54 | 17.35 | 0.250 | 0.065 | 0.60 | 78% |

![cost and iterations](plots/p7_cost.png)

![where an iteration goes](plots/p7_iteration.png)

Iterations of the first corrector of every step (`R` = rebuild of the coarse basis):

- `bub192_ref` (exact): 19 20 21 24 29 37 51 19R 19 20 23 27 19R 19 19 23 29 18R 18 19
- `bub192_sa` (amg sa+gs): 20 20 21 24 29 19R 19 19 21 26 32 19R 19 19 23 29 19R 19 20 25
- `bub256_ref` (exact): 21 21 22 28 39 21R 21 22 25 32 21R 22 22 25 33 21R
- `bub256_sa` (amg sa+gs): 22 22 22 28 39 22R 22 22 25 32 43 22R 22 22 26 34
- `bub256_sp` (exact, single): 23 23 24 30 40 23R 23 24 27 33 43 23R 23 24 28 35

