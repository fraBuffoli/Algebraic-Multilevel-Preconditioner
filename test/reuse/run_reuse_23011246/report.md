# Setup reuse (P1): run_reuse_23011246

`NP=240 COARSE='--coarse-solver cpardiso --coarse-procs 48 --coarse-root-below 10000' STEPS=60 STEPS_OFF=15 CORR=3 N_SEQ=128 N_LARGE=192`

Times are wall times between barriers, summed over the solves of each run. **cost/solve** = (initial setup + updates + solves) / solves; **speedup** is relative to the `off` run of the same group (full setup before every solve).

| run | np | solves | reuse / refactorize / rebuild | initial setup [s] | mean refactorize [s] | mean rebuild [s] | mean solve [s] | its avg / max | n0 | cost/solve [s] | speedup |
|:--|--:|--:|:--:|--:|--:|--:|--:|--:|--:|--:|--:|
| bub128_r1000_off | 240 | 45 | 0 / 0 / 44 | 4.471 | 0.000 | 3.257 | 0.189 | 12.3 / 13 | 6015 | 3.472 | 1.0x |
| bub128_r1000_refactor | 240 | 180 | 120 / 59 / 0 | 3.324 | 0.411 | 0.000 | 0.462 | 30.5 / 47 | 6015 | 0.615 | 5.6x |
| bub128_r1000_auto | 240 | 180 | 120 / 53 / 6 | 3.318 | 0.424 | 3.301 | 0.358 | 20.6 / 40 | 6015 | 0.611 | 5.7x |
| bub128_r1000_ras | 240 | 30 | 20 / 9 / 0 | 0.134 | 0.067 | 0.000 | 4.802 | 794.5 / 953 | 0 | 4.827 | 0.7x |
| bub128_r10_refactor | 240 | 180 | 120 / 59 / 0 | 3.356 | 0.472 | 0.000 | 0.233 | 13.7 / 15 | 6176 | 0.407 | - |
| bub128_r10_auto | 240 | 180 | 120 / 57 / 2 | 3.398 | 0.485 | 3.796 | 0.313 | 13.8 / 16 | 6176 | 0.528 | - |
| bub192_r1000_off | 240 | 15 | 0 / 0 / 14 | 16.049 | 0.000 | 15.315 | 0.662 | 13.0 / 13 | 12589 | 16.025 | 1.0x |
| bub192_r1000_auto | 240 | 90 | 60 / 26 / 3 | 15.233 | 2.004 | 15.459 | 1.355 | 20.7 / 39 | 12589 | 2.620 | 6.1x |
| cd100_pulse_off | 240 | 45 | 0 / 0 / 44 | 1.207 | 0.000 | 1.157 | 0.086 | 13.4 / 14 | 3671 | 1.245 | 1.0x |
| cd100_pulse_auto | 240 | 180 | 120 / 59 / 0 | 1.214 | 0.158 | 0.000 | 0.110 | 13.4 / 14 | 3671 | 0.169 | 7.4x |

## Rebuild steps and aging of the coarse basis

Iterations of the first corrector of every step (the matrix has just changed). `R` marks the steps where `auto` rebuilt the coarse basis.

- `bub128_r1000_off`: 13 12 12 12 12 12 13 12 12 12 13 12 12 12 12
- `bub128_r1000_refactor`: 13 12 14 17 19 24 28 32 36 38 40 43 47 44 44 43 40 37 35 35 33 30 31 32 32 33 32 31 31 29 29 28 28 28 28 28 29 28 29 29 30 30 32 32 32 30 30 31 30 30 29 29 29 29 28 28 27 27 27 28
- `bub128_r1000_auto`: 13 12 14 17 19 24 28 32 36 12R 13 14 17 19 21 26 33 39 12R 13 13 16 18 22 26 29 31 12R 12 13 14 17 20 24 27 30 35 12R 12 13 16 21 24 32 38 13R 13 13 16 19 24 28 33 38 12R 13 14 16 19 22
- `bub128_r1000_ras`: 791 926 801 787 718 794 662 659 893 921
- `bub128_r10_refactor`: 12 12 12 12 13 13 14 14 14 14 15 15 15 15 15 15 15 14 14 14 14 14 14 14 14 14 13 14 13 13 13 13 13 13 14 13 14 14 14 14 14 14 14 14 14 14 14 13 13 14 13 13 14 14 14 14 14 14 14 14
- `bub128_r10_auto`: 12 12 12 12 13 13 14 14 14 14 15 15 15 15 15 15 15 14 14 14 14 14 14 14 14 14 13 14 13 13 13 13 13 13 14 13 14 14 14 14 12R 12 12 12 13 13 14 14 14 15 15 16 16 16 15 16 15 15 12R 12
- `bub192_r1000_off`: 13 13 13 13 13
- `bub192_r1000_auto`: 13 13 14 16 19 23 26 30 34 13R 13 14 17 20 24 26 29 34 37 13R 13 14 17 20 24 29 33 13R 13 14
- `cd100_pulse_off`: 13 13 13 13 14 13 14 13 14 14 14 13 14 13 13
- `cd100_pulse_auto`: 13 13 13 13 13 13 13 13 13 13 14 14 14 14 13 13 14 14 14 14 13 13 13 13 13 13 13 13 13 13 14 14 14 13 13 14 14 14 14 14 13 13 13 13 13 13 13 13 13 13 13 14 14 14 14 14 14 14 14 14

## Where the update time goes

Means over the solves of each level (from the per-solve CSV). For the split inside a refactorization see the `update.*` timers at the end of `log.<run>`: `update.subdomain` (new values, ghost exchange), `update.local_refactorization` (numerical LU of A_ii), `update.coarse_assembly` (A_00 = V^T A V with the stored W), `update.coarse_factorization` (numerical LU of A_00).

- `bub128_r1000_off`: rebuild 3.2566 s (44)
- `bub128_r1000_refactor`: reuse 0.0005 s (120), refactorize 0.4115 s (59)
- `bub128_r1000_auto`: reuse 0.0004 s (120), refactorize 0.4242 s (53), rebuild 3.3015 s (6)
- `bub128_r1000_ras`: reuse 0.0011 s (20), refactorize 0.0672 s (9)
- `bub128_r10_refactor`: reuse 0.0005 s (120), refactorize 0.4722 s (59)
- `bub128_r10_auto`: reuse 0.0008 s (120), refactorize 0.4849 s (57), rebuild 3.7958 s (2)
- `bub192_r1000_off`: rebuild 15.3148 s (14)
- `bub192_r1000_auto`: reuse 0.0014 s (60), refactorize 2.0036 s (26), rebuild 15.4587 s (3)
- `cd100_pulse_off`: rebuild 1.1574 s (44)
- `cd100_pulse_auto`: reuse 0.0003 s (120), refactorize 0.1577 s (59)

