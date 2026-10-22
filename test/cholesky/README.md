# P2 — symmetric factorizations and single precision (G100 campaign)

`Allrun_p2.sh` measures what P2 changes on the problems of the P1 campaign, with the P1
configuration as the reference: only the factorizations change, everything else (problem,
decomposition, coarse space, `--reuse auto`) is the same.

```bash
cd test/p2
sbatch Allrun_p2.sh                  # 5 nodes x 48 = 240 processes, < 1 h (estimate ~35 min)
python3 report_p2.py run_p2_<jobid>  # -> run_p2_<jobid>/report.md
SMOKE=1 NP=4 ./Allrun_p2.sh          # local check of the script (tiny sizes, ~2 min)
```

## Variants

| variant | options | A_ii | Ã_ii + σI | A_00 |
|---|---|---|---|---|
| `lu` (= P1) | `--local-factorization lu --coarse-factorization lu` (root) / `ldlt` (cpardiso) | LU | LU | root: LU; cpardiso: LDLᵀ (mtype −2, what P1 used with Intel MPI) |
| `chol` | defaults (`auto` / `auto`) | Cholesky | Cholesky | root: Cholesky; cpardiso: mtype 2 + probe solve |
| `cholsp` | `chol` + `--local-precision single` | Cholesky, float | Cholesky | as `chol` |
| `chol_cp` | `chol` + `--coarse-root-below 0` | Cholesky | Cholesky | cpardiso (instead of root) |
| `chol_root` | `chol` + `--coarse-root-below 1000000` | Cholesky | Cholesky | root (instead of cpardiso) |

## Runs (240 processes)

| group | runs | what it answers | estimate |
|---|---|---|---|
| check | build, `test_subdomain`, `test_preconditioner`, `test_reuse`, `test_factorization` on 16 processes | correctness on G100; `test_factorization` is the only test of the distributed Cholesky with Intel MPI | ~7 min |
| `bub128_*` | 128³, ρ = 1000, n0 ≈ 6000 → A_00 on rank 0; 30 steps × 3 correctors: `lu`, `chol`, `cholsp`, `chol_cp` | P1 spent half of the time in the root A_00 (solves + refactorizations): how much does Cholesky remove? Is cpardiso better than root with Cholesky? | ~7 min |
| `bub192_*` | 192³ (~30k cells per subdomain), n0 ≈ 12600 → cpardiso; 30 steps: `lu`, `chol`, `cholsp`, `chol_root` | larger subdomains: local solves and refactorizations weigh more; distributed Cholesky vs LDLᵀ; root vs cpardiso | ~16 min |
| `<matrix>_*` | one solve (setup + GMRES) of G3_circuit, thermal2, ecology2 (paper settings, cpardiso): `lu`, `chol` | single solves where the cpardiso factorization of A_00 is 30-50% of the setup; skipped if `matrices/<name>.mtx` is missing | ~5 min |

If `test_factorization` fails, the runs go on with `--coarse-factorization ldlt` (the distributed
A_00 as in P1), so the local Cholesky is still measured; `config.txt` records it.

## What to look at in `report.md`

1. **Correctness first.** All runs `ok`, every solve converged, max true residual ≤ 1.1e-8; the
   factorization columns must say `cholesky` for `chol` (A_ii, Ã_ii and A_00), `cholesky single`
   for A_ii in `cholsp`, `cpardiso mtype 2 cholesky` for the distributed A_00, and no `(fallback)`:
   these matrices are SPD, a fallback would mean a factorization failed.
2. **Cost of each operation** (second table): the clean comparison, independent of the rebuild
   steps chosen by `auto`. Expected for `chol` vs `lu`: factorizations and refactorizations of
   A_ii, Ã_ii and the root A_00 about −50%; A_ii solve per iteration about −20%; root A_00 solve
   −15/−20% (part of it is the Gatherv/Scatterv); cpardiso (mtype 2 vs −2) refactorization −25/−30%.
3. **cost/solve vs lu** (first table): what a user sees. Rough expectation from the P1 timings:
   `bub128_chol` ≈ −20%, `bub192_chol` ≈ −10/−15%.
4. **`cholsp`**: the A_ii solve per iteration should drop by ~45% vs `chol`, but the iterations go up
   (+15-20% locally) and every extra iteration also costs a coarse solve: at 128³ (coarse-dominated)
   it is probably slower than `chol`, at 192³ about break-even. If so, single precision stays an
   option, not a default.
5. **`chol_cp` / `chol_root`**: which coarse solver to use at n0 ≈ 6000 and ≈ 12600 once the
   factorization is Cholesky: this sets `--coarse-root-below` for the next campaigns.

The report also lists the iterations of the first corrector of every step (`R` = rebuild) and the
refactorizations that fell back to a full factorization, as in P1.
