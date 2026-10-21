# Setup reuse (P1) on G100

Sequences of solves with the same sparsity pattern, as the pressure equation of a CFD time loop:
`--steps` time steps, `--correctors` solves per step with the same matrix and a new right-hand side.
Before every solve the preconditioner is brought up to date with one of three levels
(see the main README, *Setup reuse between solves*):

- **reuse**: the matrix did not change (correctors of one step): nothing to do;
- **refactorize**: numerical LU of A_ii, A_00 = R_0 A R_0^T with the *old* coarse basis, numerical LU of A_00;
- **rebuild**: as a new setup (eigenproblems, new coarse basis), on the existing subdomain structure.

and three modes are compared:

| `--reuse` | levels | meaning |
|---|---|---|
| `off` | rebuild before every solve | no persistent state (reference) |
| `refactor` | reuse / refactorize, never rebuild | how fast the coarse basis ages |
| `auto` | reuse / refactorize / rebuild at break-even | rebuild when the time lost to the old basis reaches the extra cost of a rebuild |

```bash
cd test/reuse
sbatch Allrun_reuse.sh                    # one job, 5 nodes, < 1 h: build, unit tests, runs
python3 plot_reuse.py run_reuse_<jobid>   # report.md + plots/<group>.png
```

| Step | Runs | What it shows |
|---|---|---|
| `RUN_CHECK` | `test_subdomain`, `test_preconditioner`, `test_reuse` on 16 processes | the reuse is exact (refactorization = fresh setup, coarse operator of the new matrix); if `test_reuse` fails the job stops |
| `RUN_SEQ` | `bubbles3d` 128^3 (~9k cells/process), density ratio 1000: `off` (15 steps), `refactor`, `auto` (60 steps) x 3 correctors | cost per solve with and without reuse, aging of the coarse basis, rebuilds chosen by `auto` |
| `RUN_CONTRAST` | same at density ratio 10 (`refactor`, `auto`); one-level RAS at 1000 | the basis ages faster with a high contrast; what the coarse space is worth |
| `RUN_LARGE` | `bubbles3d` 192^3 (~30k cells/process, as fig4 at N = 240): `off` (5 steps), `auto` (30) | gain when the setup is expensive |
| `RUN_NONSYM` | `convdiff3d` 100^3 with pulsating convection: `off`, `auto` | nonsymmetric numerical refactorization (PARDISO LU with scaling/matching from the first analysis) |

Every run leaves `log.<run>` (one line per solve, summary, timers) and `csv/<run>.csv` (one line per
solve: step, corrector, level, update and solve time, iterations, iterations per decade, n0, time
wasted by the old basis). `status.txt` lists outcome and wall time of each run. Sizes, steps,
coarse solver and number of processes are variables at the top of the script (they can also be set
from the environment, e.g. `NP=480 sbatch --nodes=10 Allrun_reuse.sh`).

**What to look at in `report.md`.**
- *cost/solve* and *speedup*: mean wall time per solve, initial setup included, relative to `off`.
  `off` pays a full setup at every solve, so its cost per solve is stationary and a few steps suffice.
- *mean refactorize* vs *mean rebuild*: the price of the two update levels on this machine and size.
  The `update.*` timers in `log.<run>` split a refactorization into new values (`update.subdomain`),
  numerical LU of A_ii, A_00 with the stored W, numerical LU of A_00.
- *Rebuild steps and aging*: iterations of the first corrector of every step. With `refactor` they
  grow as the bubbles move away from where the coarse basis was computed, then reach a plateau
  (the part of the coarse space that does not depend on the bubble positions); `auto` rebuilds
  (`R`) when the accumulated loss pays for a rebuild.

**Local smoke test** of the script (tiny problems, any MPI, no modules):
`SMOKE=1 NP=4 ./Allrun_reuse.sh` (about 1.5 min on a laptop).
