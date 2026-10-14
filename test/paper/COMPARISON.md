# Comparison with the paper

Results of `Allrun.sh` on G100 against Al Daas, Jolivet, Rees, *Efficient algebraic two-level
Schwarz preconditioner for sparse matrices*, SISC 2023 (Section 4).

- `run_22838900`: A_00 gathered and factorized on rank 0, nev = 300 eigenpairs always computed.
- `run_22843541`: A_00 factorized by MKL `cluster_sparse_solver` on 48 processes, adaptive
  number of computed eigenpairs (`--nev-start 32`).

Numbers below come from `run_22843541` unless stated. On every run that completed in both
jobs, n0 and the iteration count are identical: the changes between the two jobs make the
setup faster without changing the preconditioner.

Settings that differ from the paper:

- N = 240 subdomains instead of 256 (Table 1) or 1024 / 4096 (Table 2), so compare n0/N, not n0;
- METIS instead of ParMETIS;
- MKL instead of MUMPS for A_00, Spectra (Arnoldi) instead of SLEPc;
- for (4.1), upwind finite volumes on a uniform grid instead of SUPG finite elements on a
  boundary-layer mesh.

Common to both: right-preconditioned GMRES(30), rtol 1e-8, zero initial guess, random right-hand
side for the SuiteSparse matrices.

## In short

| Experiment | Outcome |
|---|---|
| Fig. 2, G3_circuit vs tau | reproduced: n0 within 15 % (26 % at tau = 0.6), iterations 10-30 % lower, same trend |
| Table 1, SuiteSparse | reproduced at tau = 0.9 (see the note below): n0/N within 15 % of the paper on 8 of 10 matrices, iterations within +-1 on 6 |
| Fig. 4, strong scaling 3D | reproduced: iterations bounded in N (16 -> 12 here, 28 -> 15 in the paper) |
| Table 2, convection-diffusion | 2D and 3D with nu >= 1e-2 reproduced; 3D with nu <= 1e-3 converges with a smaller coarse space and more iterations |
| Failures | finan512 at tau = 0.6 with the distributed coarse solver (crash), circuit5M_dc (two-level setup > 20 min), fig4_N240 in run 22838900 (Arnoldi, fixed) |

## Table 1: SuiteSparse matrices (nev = 300)

**Which tau to compare.** The paper states tau = 0.6, but its Table 1 row for G3_circuit
(8 it., n0 = 71,385) is the tau = 0.9 point of its own Fig. 2; at tau = 0.6 its Fig. 2 gives
11 it. and n0 = 34,402. Our tau = 0.9 column also matches the paper's n0/N on almost every matrix.
The paper's Table 1 is therefore compared with our tau = 0.9 runs.

| matrix | paper its | paper n0/N | ours tau=0.9 its | ours tau=0.9 n0/N | ours tau=0.6 its | ours tau=0.6 n0/N | one-level RAS its |
|:--|--:|--:|--:|--:|--:|--:|--:|
| finan512 | 6 | 10 | 6 | 22 | 6 (*) | 0.45 (*) | 6 |
| nxp1 | 20 | 77 | 29 | 95 | 42 | 59 | n.c. |
| CoupCons3D | 20 | 113 | 21 | 98 | 23 | 29 | 23 |
| tmt_unsym | 6 | 129 | 5 | 128 | 8 | 54 | n.c. |
| ecology2 | 6 | 133 | 5 | 130 | 8 | 56 | n.c. |
| thermal2 | 26 | 157 | 5 | 139 | 7 | 68 | n.c. |
| atmosmodj | 7 | 298 | 14 | 286 | 71 | 2 | 79 |
| G3_circuit | 8 | 279 | 7 | 279 | 10 | 106 | n.c. |
| Transport | 9 | 300 | 8 | 300 | 8 | 273 | n.c. |
| memchip | 36 | 226 | 8 | 224 | 13 | 164 | 44 |
| circuit5M_dc | 7 | 34 | - | - | timeout | - | 10 |

(*) run_22838900, A_00 on rank 0; the same run with the distributed coarse solver crashes (see below).
n.c. = not converged in 200 iterations.

**Where the coarse space matches the paper.** At tau = 0.9, n0/N is within 15 % of the paper
except for:
- nxp1: +23 %;
- finan512: 22 vs 10, where the coarse space is tiny in any case.

**Where the iterations differ:**
- nxp1 (29 vs 20) and atmosmodj (14 vs 7) take more iterations with a similar coarse space;
- thermal2 (5 vs 26) and memchip (8 vs 36) take fewer.

The coarse space does not explain either deviation, since n0/N matches. For nonsymmetric
matrices, the GMRES iteration count also depends on:
- the partition (METIS on 240 parts vs ParMETIS on 256);
- the right-hand side (random in both, but not the same vector).

This was not investigated further.

**atmosmodj and the choice of tau.** At tau = 0.6, atmosmodj selects almost nothing (n0/N = 2,
71 it.); at tau = 0.9 it selects 286 per subdomain. So about 280 of the largest local eigenvalues of
each subdomain lie between 1/0.9 = 1.11 and 1/0.6 = 1.67. On such a matrix the size of the coarse space jumps with tau,
which is consistent with the paper's Table 1 having been run with the larger threshold.

In every case where one-level RAS does not converge, the two-level method does.

## Fig. 2: G3_circuit, influence of tau (nev = 300)

| tau | n0 | paper n0 | its | paper its |
|--:|--:|--:|--:|--:|
| 0.02 | 853 | 739 | 53 | 75 |
| 0.05 | 1936 | 1831 | 34 | 44 |
| 0.1 | 3545 | 3608 | 26 | 31 |
| 0.3 | 10052 | 11784 | 16 | 18 |
| 0.6 | 25473 | 34402 | 10 | 11 |
| 0.9 | 66910 | 71385 | 7 | 8 |
| one-level | - | - | n.c. | n.c. |

## Table 2: convection-diffusion (4.1), tau = 0.3, nev = 60

Same subdomain size as the paper (~6k unknowns in 2D, ~2k in 3D) on N = 240.

| dim | nu | its | paper its | n0/N | paper n0/N | one-level RAS its |
|:--|--:|--:|--:|--:|--:|--:|
| 2D | 1 | 11 | 23 | 31 | 52 | 685 |
| 2D | 1e-1 | 11 | 20 | 31 | 52 | 892 |
| 2D | 1e-2 | 11 | 19 | 31 | 52 | n.c. (1000) |
| 2D | 1e-3 | 12 | 20 | 27 | 46 | n.c. (1000) |
| 2D | 1e-4 | 16 | 21 | 15 | 28 | n.c. (1000) |
| 3D | 1 | 12 | 18 | 11 | 44 | 90 |
| 3D | 1e-1 | 12 | 14 | 10 | 44 | 108 |
| 3D | 1e-2 | 14 | 11 | 10 | 39 | 233 |
| 3D | 1e-3 | 21 | 16 | 5 | 24 | 455 |
| 3D | 1e-4 | 97 | 29 | 3 | 19 | n.c. (1000) |

**Same trend as the paper.** In both codes, n0 decreases as nu decreases, and the iterations
stay nearly constant in 2D.

**Systematically smaller coarse space.** Ours is smaller at every nu: 0.55-0.6 times the
paper's in 2D and 0.15-0.25 times in 3D. This already holds at nu = 1, where convection plays
no role, so the cause is the discretization, not the Péclet number.

A plausible mechanism follows from how the local splitting is built. A~_ii differs from A_ii
only on the overlap rows: each row's diagonal is reduced by the absolute sum of its couplings to
unknowns outside the subdomain.
- Our 5-point (2D) and 7-point (3D) stencils couple each overlap unknown to fewer outside
  unknowns than a finite element stencil on a simplicial mesh.
- So A~_ii is closer to A_ii, and fewer eigenvalues of (3.1) exceed 1/tau.
- The gap between the stencils is larger in 3D than in 2D, consistent with the larger ratio there.

**Impact on convergence.** With n0/N >= 10 the smaller coarse space costs nothing: 12-14
iterations vs the paper's 11-18. Below that it does: 21 vs 16 at nu = 1e-3 (n0/N = 5) and
97 vs 29 at nu = 1e-4 (n0/N = 3). The two-level method still converges where one-level RAS does
not. The method's own control is tau: a larger threshold enlarges the coarse space. That run
has not been done.

## Fig. 4: strong scaling, 3D (4.1), nu = 1e-2, tau = 0.3, nev = 60

Ours: 200^3 = 8.0e6 unknowns (finite volumes). Paper: 8.1e6 unknowns (P2 elements).

| N | n0 | its | total [s] | speedup vs N = 30 |
|--:|--:|--:|--:|--:|
| 30 | 1805 | 16 | 332 | 1.0 |
| 60 | 3616 | 14 | 182 | 1.8 |
| 120 | 7126 | 13 | 69.3 | 4.8 |
| 240 | 12827 | 12 | 36.6 | 9.1 |

Paper, N = 32, 64, 128, 256, 512: 28, 21, 19, 20, 15 iterations.

**Iterations.** They stay bounded as N grows, as in the paper; ours are lower in absolute terms
because the discretization differs.

**Scaling.** The speedup is superlinear (9.1 for 8 times the processes) because the local LU
and the local eigensolver cost grow faster than linearly with the subdomain size.

**Timings.** Times are not compared with the paper: the machines and the software stacks differ.

## Failures: where and why

### finan512, tau = 0.6, coarse solver `cpardiso` (run 22843541): crash

**Where.** During setup, after the subdomains are built and before the setup summary,
processes abort with glibc's `malloc(): invalid size (unsorted)` (5 messages in the log): the
allocator detected a corrupted heap. The job exits with code 255 after 18 s.

**Why, by elimination.** The failure needs both the distributed coarse solver and a tiny n0:
- the same matrix, partition and tau with A_00 on rank 0 (run 22838900) converges: n0 = 109,
  6 iterations;
- the same coarse solver at tau = 0.9 (n0 = 5309) converges in 6 iterations;
- the local eigensolver takes the same dense LAPACK path on all 240 subdomains (~480 unknowns
  each) at both tau.

The failing combination is therefore MKL `cluster_sparse_solver` with a 109 x 109 coarse matrix
distributed over 48 processes, about 2 rows each. That is a degenerate configuration: factorizing
the matrix on one process takes microseconds.

**What is not established.** The crash was not reproduced locally (OpenMPI and Intel MPI, fewer
processes). Whether the bad write happens inside MKL or in the redistribution of such small row
blocks is not known.

**Workaround.** Use `--coarse-solver root` when n0 is small.

### circuit5M_dc, two-level (both runs): timeout after 1200 s

**Where.**
- The subdomains are small and regular: n_I = 14,256-15,119, 43-371 overlap unknowns,
  nnz(A_ii) ~ 62k.
- One-level RAS converges in 10 iterations, with 0.12 s of setup and 0.05 s of solve.
- In the two-level run (with `--verbose 2`), no rank printed `eigensolver done` within 1200 s.

So all 240 ranks spend the whole time in the local eigensolver, on a problem of ~15k unknowns.
That eigensolver has three phases:
1. the LU of A~_ii + sigma I;
2. the detection of the kernel of A~_ii;
3. Arnoldi.

Phase 1 is very unlikely to be the problem: A~_ii has the same sparsity pattern as A_ii (only
the diagonal of the overlap rows changes), and the LU of A_ii takes < 0.1 s in the RAS run.
The time is therefore spent in the kernel detection or in Arnoldi.

**Why (hypothesis, not verified).** circuit5M_dc is a nodal-analysis matrix of a circuit
(DC operating point).
- The local splitting removes the conductances to nodes outside the subdomain: a Neumann-type
  local matrix.
- Every connected group of subdomain nodes with no path to ground inside the subdomain is then
  floating, and the constant vector on that group lies in ker A~_ii.
- If the METIS parts of this circuit graph contain many such groups, dim ker A~_ii is large.

A large kernel makes the kernel detection (and the construction of (L ∩ K)^⊥) expensive, and
leaves a large cluster of infinite eigenvalues in (3.1), which slows Arnoldi. The paper reports
7 iterations with n0/N = 34 for this matrix (SLEPc, ParMETIS), so the limitation is in this
implementation, not in the method.

**How to confirm.** Confirming the hypothesis needs either a timer per phase of the local
eigensolver or an offline run on one subdomain; neither has been done. Matrices of this kind
(network graphs with floating parts) are far from the PDE matrices this code targets.

### fig4_N240 (run 22838900): Arnoldi failure, fixed

**Where.** Rank 61 reported
`local eigensolver failed: UpperHessenbergSchur: Schur decomposition failed`. Inside Spectra,
the QR iteration on the small projected Hessenberg matrix did not converge within Eigen's
iteration limit, and the whole setup was aborted.

**Fix.** The current code restarts Arnoldi after such a failure. In run 22843541 the same case
converges: n0 = 12,827, 12 iterations.

### Not failures

**One-level RAS.** The RAS runs that do not converge are expected: one-level methods do not
scale with N. The paper's Fig. 2 shows one-level ASM not converging on G3_circuit, as here.

**`error` in run 22838900.** In that job's `status.txt` these runs are marked `error` only
because GMRES returns exit code 2 on non-convergence. Run 22843541 classifies them as
`not_converged`.
