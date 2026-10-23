# P7 — inexact local solvers (G100 campaigns)

Results of the second campaign (P7b, final code): [`RESULTS.md`](RESULTS.md).

**Idea.** After P5 (`--nev 8`, `--eig-tol 1e-4`) the solves with the exact factors of A_ii are
the largest cost of a GMRES iteration on large subdomains (240 processes, 192³, ~30k cells per
process: 60-70% of an iteration), and every refactorization of a time loop is a numerical Cholesky
factorization. `--local-approx amg` replaces A_ii⁻¹ in the one-level part with one AMG V-cycle
(AMGCL, Gauss-Seidel smoother). The coarse space (Ã_ii factorized exactly) and A_00 do not change.
Theory: a B_i spectrally equivalent to A_ii⁻¹ keeps the convergence of exact local solves up to a
constant. See the P7 section of the main README.

## First campaign (`run_p7_23018578`: 240 processes, `run_p7_23019000`: 960 processes)

Run with the first P7 version (10 local solvers: exact, exact single, incomplete Cholesky, AMG with
sa / rs coarsening and spai0 / Gauss-Seidel / Chebyshev smoothers, float, two sweeps, hierarchy
reused or not). Its script is not kept: the options it used were removed afterwards. Findings:

- 240 processes: AMG `sa+gs` -47% (192³, ~30k cells per process) and -54% (256³, ~70k) in cost per
  solve against exact Cholesky, the first iterations after a rebuild equal to the exact ones;
  incomplete Cholesky +59% / +81% (its iterations grow with the subdomain size); spai0, Chebyshev,
  two sweeps equal or worse than one Gauss-Seidel sweep; Ruge-Stuben with the iterations of the
  exact solves but a heavier setup.
- Reusing the AMG hierarchy in the refactorizations made the iterations grow faster between two
  rebuilds of the coarse space; full setups followed the exact solves step by step.
- 960 processes: the coarse solve (6.7 ms per iteration) dominates, small gains; at ~17k cells per
  process the AMG hierarchy had 2 levels, with a coarsest level of 1000-3000 unknowns solved by a
  nearly dense skyline LU: setup 0.55 s (exact Cholesky 0.23 s), slow V-cycle.
- Two 960-process runs (`bub192_sp`, `bub256_sa`) were spoiled by a ~30 s stall of one PARDISO
  factorization on one rank: the reuse policy took it as the cost of a rebuild and stopped
  rebuilding (up to 109 / 131 iterations per solve).

Corrections in the code after it (P7b): only AMG with Gauss-Seidel is left (`--amg-coarsening
sa|rs`); `--amg-coarse-enough 500`; `--amg-reuse 0` by default; the reuse policy uses the lower
median of the measured rebuild / refactorization times and `--rebuild-growth 4` by default.

## Second campaign: `Allrun_p7b.sh` (240 processes) and `Allrun_p7b_960.sh` (960 processes)

```bash
cd test/p7
sbatch Allrun_p7b.sh                       # 5 nodes = 240 processes, ~35 min (limit 1 h), ~140 core-hours
sbatch Allrun_p7b_960.sh                   # 20 nodes = 960 processes, ~12 min (limit 30 min), ~200 core-hours
python3 report_p7.py run_p7b_<jobid>       # -> report.md + plots/p7_cost.png, plots/p7_iteration.png
SMOKE=1 NP=4 ./Allrun_p7b.sh               # local check of the script (tiny sizes, ~2 min)
```

Same problems as the first campaign with fewer steps (20 at 192³, 16 at 256³, 3 correctors each,
`--reuse auto`, `--tau 0.3 --nev 8 --eig-tol 1e-4`) and fewer variants: about 30% of its cost.
The 240-process job first runs `test_inexact` and `test_reuse` on 16 processes (new code); the
960-process job only builds `schwarz2lvl` and runs the time loops.

| variant | A_ii in the one-level part | 240 procs: 192³ / 256³ | 960 procs: 192³ / 256³ |
|---|---|:-:|:-:|
| `ref` | exact Cholesky (PARDISO), double | ✓ / ✓ | ✓ / ✓ |
| `sp` | exact Cholesky, single precision | ✓ / ✓ | – / ✓ |
| `sa` | AMG smoothed aggregation + GS (new defaults) | ✓ / ✓ | ✓ / ✓ |
| `rs` | AMG Ruge-Stuben + GS (new defaults) | ✓ / ✓ | – / – |
| `sasp` | `sa` in float | ✓ / ✓ | – / – |
| `sare` | `sa` with the hierarchy reused (old default) | – / ✓ | – / – |

Questions:

1. Does `sa` keep -47% / -54% at 240 processes with full setups, and is `sare` worse at 256³ too
   (the first campaign tested the reuse only at 192³)?
2. With `--amg-coarse-enough 500`, does `rs` lose its heavy setup at 192³ (0.77 s in the first
   campaign, with a coarsest level of ~2.5k unknowns) and become competitive with `sa`?
3. At 960 processes and 256³ (~17k cells), does `sa` now beat the exact solves (first campaign:
   +7%, AMG setup 0.55 s)? At 192³ (~7k cells) the gain can only be small: the coarse solve dominates.
4. Reuse policy: no run should stop rebuilding (iterations per solve should stay below ~3x the
   first one after a rebuild).

`VARIANTS_A`, `VARIANTS_B`, `STEPS_A`, `STEPS_B`, `BASE` override the defaults (e.g.
`VARIANTS_B="ref sa2" sbatch Allrun_p7b.sh` for two Gauss-Seidel sweeps).
