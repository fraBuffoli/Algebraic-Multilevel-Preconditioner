# P5 — small coarse space and fast coarse solve (G100 campaign)

**Idea.** Every GMRES iteration solves the coarse problem A_00 y = c once, and while it is
solved most ranks wait. Its cost grows with n0 = Σ m_i (m_i = coarse vectors of subdomain i):
the factorization of A_00 roughly as m_i³ at fixed N, a coarse solve as m_i², the products with
V (restrict/prolong) as m_i. In P1 the coarse solves and factorizations took ~48% of the time at
128³ and ~30% at 192³, where m_i reached the cap `--nev 60` on some subdomains. Fewer vectors make
every coarse operation cheaper and the eigensolver too (Arnoldi computes `nev` eigenpairs), but
the coarse space gets weaker and GMRES needs more iterations: the total time has an optimum.

**No code change is needed**: m_i is set by existing options (`--tau`, `--nev`), and so is
where A_00 is solved (`--coarse-solver`, `--coarse-procs`, `--coarse-root-below`). This campaign
measures where the optimum is on the bubbles problems of P1/P2 and which coarse solver to use.

```bash
cd test/p5
sbatch Allrun_p5.sh                        # 5 nodes x 48 = 240 processes, < 1 h (~37 min)
NP=960 sbatch --nodes=20 Allrun_p5.sh      # optional: where n0 = sum m_i really grows
python3 report_p5.py run_p5_<jobid>        # -> report.md + plots/p5_sweep.png
SMOKE=1 NP=4 ./Allrun_p5.sh                # local check of the script (tiny sizes, ~3 min)
```

## Runs

| group | variants | what it answers | estimate (240 procs) |
|---|---|---|---|
| `bub192_*` (sweep) | `ref` (τ 0.3, nev 60 = P1/P2), `t010` (τ 0.1), `t003` (τ 0.03), `n32`, `n16`, `n8` (nev cap); 30 steps x 3 correctors, `--reuse auto` | iterations and cost per solve as m_i shrinks, on subdomains of ~30k cells (n0 ~ 12600 at `ref`) | ~24 min |
| `bub192_cs_*` | `root`, `cpK` with K = nodes, 4 x nodes, 48; `ref` setting, 4 steps, `--reuse refactor` | where to factorize and solve A_00 at n0 ~ 12600 | ~5 min |
| `bub128_*` (sweep) | `ref`, `t010`, `n16`, `n8`; 30 steps | same at ~9k cells per subdomain (n0 ~ 6000, already on rank 0) | ~6 min |

Two ways to shrink m_i, which behave differently:

- **τ (threshold).** Only eigenvalues with |λ| ≥ 1/τ are kept: τ = 0.1 keeps |λ| ≥ 10, τ = 0.03
  keeps |λ| ≥ 33. The selection stays the one covered by the theory of the paper (a bound on the
  condition number that depends on τ, not on N or on the contrast), only weaker as τ decreases
  (paper, Fig. 2: fewer vectors, more iterations). Arnoldi still computes nev = 60 eigenpairs,
  so the eigensolver costs as much as in `ref`.
- **nev (cap).** At most nev vectors per subdomain, the largest ones. Arnoldi computes only nev
  eigenpairs (cheaper eigensolver), but a subdomain that needs more vectors (e.g. one cut by
  several bubbles) loses the guarantee.

`n0 < 10000` goes to the root coarse solver (`--coarse-root-below 10000`, as in P1/P2), so the
sweep also changes the coarse solver: the report says which one each run used.

**Coarse solver group.** `cpardiso` with K leaders: a leader owns the rows of N/K consecutive
ranks, so with K = number of nodes (5 at 240 processes) every gather/scatter of a coarse solve
stays inside a node and only 5 processes take part in the distributed factorization and solve;
K = 48 is the P1/P2 setting; `root` gathers everything on rank 0. With `--reuse refactor` all
runs do the same operations (no rebuild decisions).

## What to look at in `report.md`

1. **Correctness:** all runs `ok`, every solve converged.
2. **Sweep table and plot:** cost per solve relative to `ref`, measured and with median times
   (the median version ignores the timing bursts of the network seen in the P1 job). The
   optimum is the variant with the lowest cost; *fresh its* (first solve after a rebuild) shows
   how good the coarse space is before it ages, *its avg* includes the aging between rebuilds.
3. **Where an iteration goes:** local solve, coarse solve, products with V, rest. With a small m_i
   the coarse part should shrink to a few ms and the local solve becomes the largest term.
4. **τ vs nev at similar m_i:** if the nev cap gives the same iterations as τ with a cheaper
   eigensolver, the cap is the better knob for this problem (and vice versa).
5. **Coarse solver table:** coarse solve [ms/iteration] and refactorization of A_00 for root and
   for each K: this decides the default placement of A_00 and the `--coarse-root-below` threshold.

## What comes after (depending on the results)

- If a small m_i costs few extra iterations: use it as the default for `p_rgh` (and check it at
  960 processes, where n0 grows 4x).
- If the coarse solve still dominates after shrinking m_i: code changes become worth it, e.g.
  overlapping the coarse solve with the local solves (additive variant) or a coarse solve
  replicated on one process per node; their cost is known only after this campaign.
- If iterations grow a lot already at m_i ~ 10-16: the vectors that matter are the "bubble"
  modes, which P4 (one coarse vector per bubble piece, no eigensolver) builds directly.

## Second campaign: single precision and eigensolver tolerance (`Allrun_p5b.sh`)

Result of the first campaign (`run_p5_23013515`, 240 processes): `--nev 16` or `8` halves the
cost per solve at 128³ and 192³ (cheaper coarse operations, and a 4x cheaper eigensolver that lets
`auto` rebuild more often). After that the solves with the factors of A_ii are ~45% of the time and
the eigensolver ~2/3 of a rebuild. `Allrun_p5b.sh` tries the two existing options that target them,
at 960 processes (n0 = Σ m_i grows 4x):

| variant | options | question |
|---|---|---|
| `ref` | τ 0.3, nev 60 | baseline at 960 processes (P1/P2 setting) |
| `n16`, `n8` | nev 16 / 8 | does the P5 gain hold with 4x more subdomains? |
| `n16sp`, `n8sp` | + `--local-precision single` | now that an extra iteration costs little coarse work, do float factors pay? |
| `n16e4` | + `--eig-tol 1e-4` | does a looser Arnoldi tolerance keep the coarse space (fresh iterations) and halve the eigensolver? |
| `n16spe4`, `n8spe4` | both | the expected best combination |

Problems: `bub192` (the 192³ problem of P5, ~7k cells per process: strong scaling from 240
processes) and `bub256` (16.8M cells, ~17k per process, same bubbles), 30 and 24 steps.

```bash
cd test/p5
sbatch Allrun_p5b.sh                        # 20 nodes = 960 processes, ~45 min (limit 1 h 30)
NP=1200 sbatch --nodes=25 Allrun_p5b.sh     # optional, 25 nodes
python3 report_p5.py run_p5b_<jobid>        # same report: reads run_p5b_<jobid>/variants.txt
```

Requires P2 (`--local-precision`); the script stops after the build if the option is missing.
