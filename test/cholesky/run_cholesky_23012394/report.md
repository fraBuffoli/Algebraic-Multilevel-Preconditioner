# P2 - symmetric factorizations and single precision: run_cholesky_23012394

`NP=240 COARSE='--coarse-solver cpardiso --coarse-procs 48 --coarse-root-below 10000' STEPS=30 STEPS_LARGE=30 CORR=3 N_SEQ=128 N_LARGE=192`

`not a git checkout: see src_hash.txt sources md5: b41f63a702b7`

Runs not ok: none

Unit tests: test_subdomain: ALL PASSED; test_preconditioner: ALL PASSED; test_reuse: ALL PASSED; test_factorization: ALL PASSED

## Time loops (--reuse auto)

Factorizations as reported by the run (rank 0). **cost/solve** = (initial setup + updates + solves) / solves; **vs lu** is relative to the `lu` run of the group (P1 configuration).

| run | A_ii | A~_ii | A_00 | solves (conv.) | its avg / max | refact. / rebuilds | initial setup [s] | mean refact. [s] | mean rebuild [s] | solve [ms/it] | cost/solve [s] | vs lu |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| bub128_lu | pardiso lu | pardiso lu | root pardiso lu | 90 (90) | 20.2 / 39 | 26 / 3 | 3.792 | 0.421 | 3.165 | 15.03 | 0.573 | ref |
| bub128_chol | pardiso cholesky | pardiso cholesky | root pardiso cholesky | 90 (90) | 20.2 / 39 | 26 / 3 | 2.843 | 0.233 | 2.718 | 13.92 | 0.471 | -18% |
| bub128_cholsp | pardiso cholesky single | pardiso cholesky | root pardiso cholesky | 90 (90) | 22.3 / 40 | 26 / 3 | 2.993 | 0.231 | 2.719 | 12.72 | 0.474 | -17% |
| bub128_chol_cp | pardiso cholesky | pardiso cholesky | cpardiso mtype 2 cholesky | 90 (90) | 18.1 / 33 | 26 / 3 | 4.290 | 0.256 | 2.902 | 24.50 | 0.663 | +16% |
| bub192_lu | pardiso lu | pardiso lu | cpardiso mtype -2 ldlt | 90 (90) | 22.5 / 42 | 26 / 3 | 14.596 | 1.791 | 14.624 | 54.77 | 2.400 | ref |
| bub192_chol | pardiso cholesky | pardiso cholesky | cpardiso mtype 2 cholesky | 90 (90) | 22.5 / 42 | 26 / 3 | 13.532 | 1.303 | 13.516 | 52.15 | 2.151 | -10% |
| bub192_cholsp | pardiso cholesky single | pardiso cholesky | cpardiso mtype 2 cholesky | 90 (90) | 25.5 / 47 | 27 / 2 | 13.558 | 1.234 | 13.429 | 44.18 | 1.947 | -19% |
| bub192_chol_root | pardiso cholesky | pardiso cholesky | root pardiso cholesky | 90 (90) | 20.7 / 40 | 26 / 3 | 13.463 | 1.910 | 13.326 | 63.00 | 2.454 | +2% |

## Cost of each operation

Timers averaged over the ranks, divided by the number of operations: factorizations per setup (initial + rebuilds), refactorizations per refactorize step, solves per GMRES iteration (one preconditioner application each). In brackets: change vs `lu`.

**bub128**

| operation | bub128_lu | bub128_chol | bub128_cholsp | bub128_chol_cp |
|:--|--:|--:|--:|--:|
| A_ii factorization [s] | 0.191 | 0.109 (-43%) | 0.141 (-26%) | 0.087 (-54%) |
| A_ii refactorization [s] | 0.053 | 0.028 (-48%) | 0.025 (-53%) | 0.028 (-48%) |
| A~_ii factorization [s] | 0.106 | 0.075 (-29%) | 0.076 (-28%) | 0.075 (-29%) |
| eigensolver [s] | 2.212 | 1.996 (-10%) | 1.999 (-10%) | 2.008 (-9%) |
| A_00 factorization [s] | 0.601 | 0.354 (-41%) | 0.358 (-40%) | 0.865 (+44%) |
| A_00 refactorization [s] | 0.324 | 0.170 (-48%) | 0.171 (-47%) | 0.191 (-41%) |
| A_ii solve [ms/it] | 4.47 | 3.77 (-16%) | 2.27 (-49%) | 3.81 (-15%) |
| A_00 solve [ms/it] | 7.77 | 7.42 (-5%) | 7.74 (-0%) | 17.85 (+130%) |

**bub192**

| operation | bub192_lu | bub192_chol | bub192_cholsp | bub192_chol_root |
|:--|--:|--:|--:|--:|
| A_ii factorization [s] | 0.722 | 0.445 (-38%) | 0.388 (-46%) | 0.440 (-39%) |
| A_ii refactorization [s] | 0.373 | 0.182 (-51%) | 0.141 (-62%) | 0.183 (-51%) |
| A~_ii factorization [s] | 0.577 | 0.364 (-37%) | 0.365 (-37%) | 0.365 (-37%) |
| eigensolver [s] | 9.602 | 9.083 (-5%) | 9.078 (-5%) | 9.087 (-5%) |
| A_00 factorization [s] | 2.539 | 2.280 (-10%) | 2.269 (-11%) | 2.084 (-18%) |
| A_00 refactorization [s] | 0.839 | 0.610 (-27%) | 0.600 (-28%) | 1.215 (+45%) |
| A_ii solve [ms/it] | 21.29 | 19.68 (-8%) | 11.14 (-48%) | 19.62 (-8%) |
| A_00 solve [ms/it] | 20.66 | 19.46 (-6%) | 20.70 (+0%) | 29.05 (+41%) |

Correctness: max true relative residual over all solves, and refactorizations that fell back to a full factorization.

- `bub128_chol`: max true residual 9.92e-09; local 0 (on 0 ranks), coarse 0
- `bub128_chol_cp`: max true residual 9.92e-09; local 0 (on 0 ranks), coarse 0
- `bub128_cholsp`: max true residual 9.95e-09; local 0 (on 0 ranks), coarse 0
- `bub128_lu`: max true residual 9.92e-09; local 0 (on 0 ranks), coarse 0
- `bub192_chol`: max true residual 9.86e-09; local 0 (on 0 ranks), coarse 0
- `bub192_chol_root`: max true residual 9.99e-09; local 0 (on 0 ranks), coarse 0
- `bub192_cholsp`: max true residual 9.98e-09; local 0 (on 0 ranks), coarse 0
- `bub192_lu`: max true residual 9.86e-09; local 0 (on 0 ranks), coarse 0

Iterations of the first corrector of every step (`R` = rebuild of the coarse basis):

- `bub128_chol`: 13 12 14 17 19 24 28 32 36 12R 13 14 17 19 21 26 33 39 12R 13 13 16 18 22 26 29 31 12R 12 13
- `bub128_chol_cp`: 13 12 14 17 19 24 28 32 12R 13 14 16 18 21 24 28 12R 13 14 16 19 22 25 30 12R 13 13 14 15 19
- `bub128_cholsp`: 15 14 16 19 21 27 31 34 37 14R 15 16 19 21 23 29 35 39 14R 15 15 18 20 25 29 32 33 14R 14 15
- `bub128_lu`: 13 12 14 17 19 24 28 32 36 12R 13 14 17 19 21 26 33 39 12R 13 13 16 18 22 26 29 31 12R 12 13
- `bub192_chol`: 13 13 14 16 19 23 26 30 34 42 13R 13 14 17 20 24 27 29 34 36 13R 13 13 17 20 24 28 33 38 14R
- `bub192_chol_root`: 13 13 14 16 19 23 26 30 34 13R 13 14 17 20 24 26 29 34 14R 14 14 16 19 24 28 34 39 13R 13 14
- `bub192_cholsp`: 15 15 16 18 21 25 29 33 35 42 15R 15 16 19 22 26 30 32 35 37 15R 15 15 19 22 26 31 34 38 47
- `bub192_lu`: 13 13 14 16 19 23 26 30 34 42 13R 13 14 17 20 24 27 29 34 36 13R 13 13 17 20 24 28 33 38 14R

## Single solves (setup + GMRES)

| run | A_ii | A~_ii | A_00 | n0 | its | setup [s] | eigensolver [s] | A_00 fact. [s] | solve [s] | total [s] | vs lu |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| G3_circuit_lu | pardiso lu | pardiso lu | cpardiso mtype -2 ldlt | 25473 | 10 | 13.216 | 9.324 | 3.497 | 0.293 | 13.509 | ref |
| G3_circuit_chol | pardiso cholesky | pardiso ldlt (fallback from cholesky) | cpardiso mtype 2 cholesky | 25473 | 10 | 13.093 | 9.322 | 3.375 | 0.265 | 13.358 | -1% |
| ecology2_lu | pardiso lu | pardiso lu | cpardiso mtype -2 ldlt | 13380 | 8 | 1.462 | 0.531 | 0.821 | 0.168 | 1.631 | ref |
| ecology2_chol | pardiso cholesky | pardiso cholesky | cpardiso mtype 2 cholesky | 13380 | 8 | 1.406 | 0.461 | 0.833 | 0.177 | 1.583 | -3% |
| thermal2_lu | pardiso lu | pardiso lu | cpardiso mtype -2 ldlt | 16251 | 7 | 2.381 | 1.067 | 1.149 | 0.165 | 2.546 | ref |
| thermal2_chol | pardiso cholesky | pardiso cholesky | cpardiso mtype 2 cholesky | 16251 | 7 | 2.241 | 0.964 | 1.114 | 0.142 | 2.383 | -6% |

