# Comparison with the paper

Results of `Allrun.sh` on G100 against Al Daas, Jolivet, Rees, *Efficient algebraic two-level
Schwarz preconditioner for sparse matrices*, SISC 2023 (Section 4).

- `run_22838900`: A_00 gathered and factorized on rank 0, nev = 300 eigenpairs always computed.
- `run_22843541`: A_00 factorized by MKL `cluster_sparse_solver` on 48 processes, adaptive
  number of computed eigenpairs (`--nev-start 32`).
- `run_22861729`: as `run_22843541`, plus PARDISO without iterative refinement (local and
  cluster solver), kernel detection with multi-RHS solves, and A_00 on rank 0 when n0 < 10000
  (`--coarse-root-below 10000`, crossover measured in `run_22861093` with `Allrun_perf.sh`).

Numbers below come from `run_22861729` unless stated. On every two-level run that completed in
more than one job, n0 and the iteration count are identical: the changes between the jobs make
the setup and the solve faster without changing the preconditioner (see [Performance](#performance-run-22861729-vs-run-22843541)).

Settings that differ from the paper:

- N = 240 subdomains instead of 256 (Table 1) or 1024 / 4096 (Table 2), so compare n0/N, not n0;
- METIS instead of ParMETIS;
- MKL PARDISO instead of MUMPS: sequential for the local LU, `cluster_sparse_solver` on 48
  processes for A_00 when n0 >= 10000, sequential on rank 0 below; Spectra (implicitly restarted
  Arnoldi) instead of SLEPc;
- for (4.1), upwind finite volumes on a uniform grid instead of SUPG finite elements on a
  semi-structured boundary-layer mesh (P1 in 2D, P2 in 3D: Table 2 of the paper, k = 1 and 2).

Common to both: right-preconditioned GMRES(30), rtol 1e-8, zero initial guess, random right-hand
side for the SuiteSparse matrices.

## In short

| Experiment | Outcome |
|---|---|
| Fig. 2, G3_circuit vs tau | reproduced: n0 within 15 % (26 % at tau = 0.6), iterations 9-29 % lower, same trend |
| Table 1, SuiteSparse | reproduced at tau = 0.9 (see the note below): n0/N within 15 % of the paper on 8 of 10 matrices, iterations within +-1 on 6 |
| Fig. 4, strong scaling 3D | reproduced: iterations bounded in N (16 -> 12 here, 28 -> 15 in the paper) |
| Table 2, convection-diffusion | 2D and 3D with nu >= 1e-2 reproduced; 3D with nu <= 1e-3 converges with a smaller coarse space and more iterations |
| Failures | circuit5M_dc: two-level setup > 20 min in all three jobs. Fixed or avoided: finan512 at tau = 0.6 (crash with the distributed coarse solver, run 22843541), fig4_N240 (Arnoldi, run 22838900) |

## Table 1: SuiteSparse matrices (nev = 300)

**Which tau to compare.** The paper states tau = 0.6, but its Table 1 row for G3_circuit
(8 it., n0 = 71,385) is the tau = 0.9 point of its own Fig. 2; at tau = 0.6 its Fig. 2 gives
11 it. and n0 = 34,402. Our tau = 0.9 column also matches the paper's n0/N on almost every matrix.
The paper's Table 1 is therefore compared with our tau = 0.9 runs.

| matrix | paper its | paper n0/N | ours tau=0.9 its | ours tau=0.9 n0/N | ours tau=0.6 its | ours tau=0.6 n0/N | one-level RAS its |
|:--|--:|--:|--:|--:|--:|--:|--:|
| finan512 | 6 | 10 | 6 | 22 | 6 | 0.45 | 6 |
| nxp1 | 20 | 77 | 29 | 95 | 42 | 59 | n.c. |
| CoupCons3D | 20 | 113 | 21 | 98 | 23 | 29 | 23 |
| tmt_unsym | 6 | 129 | 5 | 128 | 8 | 54 | n.c. |
| ecology2 | 6 | 133 | 5 | 130 | 8 | 56 | n.c. |
| thermal2 | 26 | 157 | 5 | 139 | 7 | 68 | n.c. |
| atmosmodj | 7 | 298 | 14 | 286 | 71 | 1.7 | 79 |
| G3_circuit | 8 | 279 | 7 | 279 | 10 | 106 | n.c. |
| Transport | 9 | 300 | 8 | 300 | 8 | 273 | n.c. |
| memchip | 36 | 226 | 8 | 224 | 13 | 164 | 44 |
| circuit5M_dc | 7 | 34 | - | - | timeout | - | 10 |

n.c. = not converged in 200 iterations. finan512 at tau = 0.6 (n0 = 109) runs with A_00 on rank 0
(`--coarse-root-below`); with `cluster_sparse_solver` it crashed in run 22843541 (see below).

**Where the coarse space matches the paper.** At tau = 0.9, n0/N is within 15 % of the paper
except for:
- nxp1: +23 %;
- finan512: 22 vs 10, where the coarse space is tiny in any case.

**Where the iterations differ:**
- nxp1 (29 vs 20) and atmosmodj (14 vs 7) take more iterations with a coarse space of similar
  or larger size (nxp1: +23 %);
- thermal2 (5 vs 26) and memchip (8 vs 36) take fewer.

The size of the coarse space does not explain either deviation. For nonsymmetric matrices, the
GMRES iteration count also depends on:
- the partition (METIS on 240 parts vs ParMETIS on 256);
- the right-hand side (random in both, but not the same vector).

This was not investigated further.

**atmosmodj and the choice of tau.** At tau = 0.6, atmosmodj selects almost nothing (n0/N = 1.7,
71 it., hardly fewer than the 79 of one-level RAS); at tau = 0.9 it selects 286 per subdomain on
average (m_i from 159 to 300, so the cap nev = 300 is reached on some subdomains). On average
about 285 of the computed eigenvalues of each subdomain therefore have modulus between
1/0.9 = 1.11 and 1/0.6 = 1.67. On such a matrix the size of the coarse space jumps with tau,
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
| 2D | 1e-1 | 11 | 20 | 31 | 52 | 890 |
| 2D | 1e-2 | 11 | 19 | 31 | 52 | n.c. (1000) |
| 2D | 1e-3 | 12 | 20 | 27 | 46 | n.c. (1000) |
| 2D | 1e-4 | 16 | 21 | 15 | 28 | n.c. (1000) |
| 3D | 1 | 12 | 18 | 11 | 44 | 90 |
| 3D | 1e-1 | 12 | 14 | 10 | 44 | 108 |
| 3D | 1e-2 | 14 | 11 | 10 | 39 | 233 |
| 3D | 1e-3 | 21 | 16 | 5 | 24 | 455 |
| 3D | 1e-4 | 97 | 29 | 3 | 19 | n.c. (1000) |

The one-level RAS run at 2D, nu = 1e-1 took 892 iterations in run 22843541 and 890 here: over
~900 iterations of GMRES(30) the round-off of the local solves (no iterative refinement since
run 22861729) changes the count by 2. The two-level runs are identical.

**Same trend as the paper.** In both codes, n0 decreases as nu decreases, and the iterations
stay nearly constant in 2D.

**Systematically smaller coarse space.** Ours is smaller at every nu: 0.53-0.60 times the
paper's n0/N in 2D and 0.16-0.25 times in 3D. This already holds at nu = 1, where the problem is
diffusion-dominated (cell Péclet number << 1), so the cause is the discretization, not the
Péclet number.

A plausible mechanism follows from how the local splitting is built. A~_ii differs from A_ii
only on the overlap rows: each row's diagonal is reduced by the absolute sum of its couplings to
unknowns outside the subdomain.
- Our 5-point (2D) and 7-point (3D) stencils couple each overlap unknown to fewer outside
  unknowns than the paper's elements: P1 triangles in 2D (about 6 neighbours per vertex) and
  P2 tetrahedra in 3D (tens of neighbours per degree of freedom).
- So A~_ii is closer to A_ii, and fewer eigenvalues of (3.1) exceed 1/tau.
- The gap between the stencils is much larger in 3D (P2) than in 2D (P1), consistent with the
  larger ratio there.

**Impact on convergence.** Where n0/N >= 10 the smaller coarse space costs nothing: in 2D
(n0/N = 15-31) 11-16 iterations vs the paper's 19-23, in 3D (n0/N ~ 10) 12-14 vs 11-18.
Below that it does: 21 vs 16 at nu = 1e-3 (n0/N = 5) and 97 vs 29 at nu = 1e-4 (n0/N = 3).
The two-level method still converges where one-level RAS does not. The method's own control is
tau: a larger threshold enlarges the coarse space. That run has not been done.

## Fig. 4: strong scaling, 3D (4.1), nu = 1e-2, tau = 0.3, nev = 60

Ours: 200^3 = 8.0e6 unknowns (finite volumes). Paper: 8.1e6 unknowns (P2 elements).

| N | n0 | its | setup [s] | solve [s] | total [s] | speedup vs N = 30 |
|--:|--:|--:|--:|--:|--:|--:|
| 30 | 1805 | 16 | 181 | 6.03 | 187 | 1.0 |
| 60 | 3616 | 14 | 94.1 | 3.39 | 97.5 | 1.9 |
| 120 | 7126 | 13 | 35.8 | 1.38 | 37.2 | 5.0 |
| 240 | 12827 | 12 | 16.1 | 0.71 | 16.8 | 11.1 |

Paper, N = 32, 64, 128, 256, 512: 28, 21, 19, 20, 15 iterations.

**Iterations.** They stay bounded as N grows, as in the paper; ours are lower in absolute terms
because the discretization differs.

**Scaling.** The speedup is superlinear (11.1 for 8 times the processes). The setup is 96-97 %
of the total, and its two largest parts grow faster than linearly with the subdomain size:
- the LU of A_ii takes 37.3 s on 267k unknowns (N = 30) and 0.97 s on 33k (N = 240), 38 times
  for 8 times the size;
- the local eigensolver takes 120 s and 11.2 s on average (each Arnoldi run applies the operator
  a similar number of times, and every application is a solve with the LU of A~_ii + sigma I).

The coarse problem stays small (n0 <= 12827): for N <= 120 it is factorized on rank 0.

**Timings.** Times are not compared with the paper: the machines and the software stacks differ.

## Failures: where and why

### finan512, tau = 0.6, coarse solver `cpardiso` (run 22843541): crash, avoided since run 22861729

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

**Now.** With `--coarse-root-below 10000` an A_00 of this size is factorized on rank 0, so the
configuration is no longer reached: run 22861729 converges in 6 iterations with n0 = 109, as
run 22838900. The cause itself was not investigated further.

### circuit5M_dc, two-level (all three jobs): timeout after 1200 s

**Where.**
- The matrix is symmetric (the log reports `block size 1, symmetric`), so only the right kernel
  of A~_ii is computed.
- The subdomains are small and regular: n_I = 14,256-15,119, 43-371 overlap unknowns,
  nnz(A_ii) ~ 62k.
- One-level RAS converges in 10 iterations, with 0.08 s of setup and 0.03 s of solve.
- In the two-level run (with `--verbose 2`), no rank printed `eigensolver done` within 1200 s,
  in run 22861729 too, where the sparse solves of the eigensolver are about 3 times faster and the
  kernel detection uses multi-RHS solves.

So all 240 ranks spend the whole time in the local eigensolver, on a problem of ~15k unknowns.
That eigensolver has three phases:
1. the LU of A~_ii + sigma I;
2. the detection of the kernel of A~_ii;
3. Arnoldi.

Phase 1 is very unlikely to be the problem: A~_ii has the same sparsity pattern as A_ii (only
the diagonal of the overlap rows changes), and the LU of A_ii takes < 0.1 s in the RAS run.
The time is therefore spent in the kernel detection or in Arnoldi.

**Why (two hypotheses, neither verified).**

*1. A large kernel.* circuit5M_dc is a nodal-analysis matrix of a circuit (DC operating point).
- The local splitting removes the conductances to nodes outside the subdomain: a Neumann-type
  local matrix.
- Every connected group of subdomain nodes with no path to ground inside the subdomain is then
  floating, and the constant vector on that group lies in ker A~_ii.
- If the METIS parts of this circuit graph contain many such groups, dim ker A~_ii is large.

In this implementation the kernel is projected out of the Arnoldi operator
(T = Pi_K (A~ + sigma I)^-1 Pi_R B Pi_K), so a large kernel does not slow Arnoldi through a
cluster of large eigenvalues. Its cost is elsewhere:
- the kernel detection doubles its probe block while every probe vector lies in the kernel, so
  it ends with p >= dim K_i columns: one multi-RHS solve, a Householder QR and an SVD of an
  n_i x p block per step;
- (L ∩ K)^⊥ needs the SVD of B Q (n_i x dim K_i);
- every Arnoldi application adds two products with the n_i x dim K_i basis Q.

Measured offline on one core, the QR and the SVD of a 15k x 1024 block take about 11 s together,
growing as p^2. To exceed 20 min this way, dim K_i must be of the order of several thousands,
a large fraction of the ~15k unknowns of a subdomain.

*2. Arnoldi stagnation.* Many identical conductances can give eigenvalues of (3.1) with high
multiplicity. When such a cluster straddles the number k of computed eigenpairs, implicit
restarts converge slowly. CoupCons3D (tau = 0.6) is a likely milder instance: one subdomain
needs 8668 operator applications, against 0.77 s of eigensolver on average. With up to 300 eigenpairs
(ncv = 601) on ~15k unknowns, each restart costs seconds of dense work, and a few hundred
restarts exceed 20 min.

Both would hit all 240 subdomains at once only if the property is shared by the whole circuit,
which is what the log shows (no rank finished). The paper reports 7 iterations with n0/N = 34
for this matrix (SLEPc, ParMETIS), so the limitation is in this implementation, not in the method.

**How to confirm.** One line per rank and per phase of the local eigensolver at `--verbose 2`
(after the factorization, after the kernel detection with dim K_i, after each Arnoldi level
with k and the operator applications) separates the two hypotheses in a single 20-minute run;
an offline run on one subdomain would too. Neither has been done. Matrices of this kind
(network graphs with floating parts) are far from the PDE matrices this code targets.

### fig4_N240 (run 22838900): Arnoldi failure, fixed

**Where.** Rank 61 reported
`local eigensolver failed: UpperHessenbergSchur: Schur decomposition failed`. Inside Spectra,
the QR iteration on the small projected Hessenberg matrix did not converge within Eigen's
iteration limit, and the whole setup was aborted.

**Fix.** The current code restarts Arnoldi after such a failure. Since run 22843541 the same case
converges: n0 = 12,827, 12 iterations.

### Not failures

**One-level RAS.** The RAS runs that do not converge are expected: one-level methods do not
scale with N. The paper's Fig. 2 shows one-level ASM not converging on G3_circuit, as here.

**`error` in run 22838900.** In that job's `status.txt` these runs are marked `error` only
because GMRES returns exit code 2 on non-convergence. Later runs classify them as
`not_converged`.

## Performance: run 22861729 vs run 22843541

Same preconditioner (n0 and iterations identical on every two-level run), with:
- PARDISO without iterative refinement, for the local solver and for `cluster_sparse_solver`
  (`iparm[7] = 0`; Eigen's PardisoLU sets 2, which performs two refinement steps, each one SpMV
  plus one solve, in every solve). PARDISO still refines when pivots were perturbed; no run
  reported perturbed pivots;
- kernel detection with one multi-RHS solve per inverse iteration step (3 steps; with 2, in
  run 22861093, one vector of nxp1 crossed `kernel_tol` and n0 grew by 1);
- A_00 on rank 0 when n0 < 10000.

Max over ranks, seconds (setup and solve as in the CSV):

| run | setup | solve | total |
|:--|--:|--:|--:|
| fig4_N30 | 316 -> 181 | 16.1 -> 6.03 | 332 -> 187 |
| fig4_N240 | 34.7 -> 16.1 | 1.92 -> 0.71 | 36.6 -> 16.8 |
| tab2_2d_nu1e-2_2L | 1.27 -> 1.19 | 0.32 -> 0.09 | 1.60 -> 1.28 |
| tab2_3d_nu1e-2_2L | 0.75 -> 0.44 | 0.32 -> 0.04 | 1.08 -> 0.49 |
| fig2_tau0.02 | 0.79 -> 0.47 | 0.89 -> 0.13 | 1.68 -> 0.60 |
| tab1_G3_circuit_2L (tau 0.6) | 14.9 -> 13.0 | 0.43 -> 0.20 | 15.3 -> 13.2 |
| tab1_G3_circuit_2L_t0.9 | 64.0 -> 61.3 | 1.33 -> 0.51 | 65.3 -> 61.8 |
| tab1_memchip_2L | 38.4 -> 36.6 | 2.70 -> 0.47 | 41.1 -> 37.1 |
| tab1_Transport_2L | 75.8 -> 63.5 | 5.78 -> 0.67 | 81.6 -> 64.2 |
| tab1_atmosmodj_2L_t0.9 | 99.3 -> 93.9 | 13.1 -> 1.46 | 112 -> 95.3 |
| tab1_nxp1_2L | 24.9 -> 22.1 | 2.79 -> 0.58 | 27.7 -> 22.7 |
| tab1_CoupCons3D_2L | 47.1 -> 34.3 | 0.73 -> 0.20 | 47.8 -> 34.6 |

`plot_paper.py run_22861729 --ref run_22843541` gives all the runs.

**Where the gain comes from** (fig4_N240, average over ranks):
- applications of the Arnoldi operator: 12.9 -> 4.3 s;
- kernel detection: 3.0 -> 0.46 s;
- dense Krylov work of Spectra (Arnoldi minus operator): 5.6 s in both;
- factorization of A_00: 1.9 s in both (`iparm[7]` only acts in the solve phase).

On the SuiteSparse matrices the solve gains up to 9 times (Transport 5.78 -> 0.67 s,
atmosmodj at tau = 0.9 13.1 -> 1.46 s), where the coarse solve with refinement dominated it. The
setup gains less there, because it is dominated by work these changes do not touch: the dense
Krylov work with nev = 300 and, for n0 > 50k, the factorization of A_00.

**Where the time goes now.**
- fig4_N240 (setup 16.1 s): local eigensolver 11.2 s on average (dense Krylov work 5.7 s,
  operator 4.3 s), coarse basis + assembly + factorization 3.3 s, LU of A_ii 1.0 s. Solve
  0.71 s: local LU 0.31 s, coarse solve 0.21 s.
- Table 1 with nev = 300: dense Krylov work (memchip: 19 of the 20 s of eigensolver) and, for
  n0 > 50k, the factorization of A_00 (G3_circuit tau = 0.9: 40 s of 61; Transport: 37 s of 64;
  atmosmodj tau = 0.9: 64 s of 94) and its solve (atmosmodj tau = 0.9: 1.3 s for 14 iterations).
- Load imbalance: CoupCons3D at tau = 0.6 spends 0.77 s in the eigensolver on average and 33.7 s
  on the slowest rank (8668 operator applications). There the coarse space does not reduce the
  iterations anyway (23, as one-level RAS).

**Coarse solver on rank 0 or distributed** (run 22861093, same problems with both; coarse solve
per GMRES iteration and factorization of A_00, max over ranks):

| n0 | root: solve/it [ms], factorization [s] | cp48: solve/it [ms], factorization [s] |
|--:|--:|--:|
| 710-3545 | 0.6-2.0, 0.01-0.13 | 5.0-7.1, 0.04-0.14 |
| 7443 | 5.8, 0.34 | 8.5, 0.38 |
| 10052 | 7.5, 0.78 | 7.8, 0.60 |
| 13380-16251 | 15-18, 1.2-1.6 | 10-11.5, 0.8-1.1 |
| 25473 | 38, 4.5 | 15, 3.4 |

The crossover is at n0 of about 10000 (N = 240, 48 processes for `cluster_sparse_solver`); it
has to be measured again for other N.

**Run-to-run variance.** The setup column is the wall time between two barriers, so it also
includes the construction of the subdomains. In tab2_3d_nu1_2L, tab2_3d_nu1e-3_2L and
tab1_ecology2_2L that phase (`setup.subdomain`) took 0.6-0.9 s instead of 0.01-0.02 s, which
makes their total setup look slower than in run 22843541 although the preconditioner setup
itself (timer `setup`) is faster, e.g. 0.68 -> 0.43 s for tab2_3d_nu1_2L. ecology2 also shows
+0.35 s in the `cluster_sparse_solver` factorization (n0 = 13380, same solver in both runs).
Runs of about one second are therefore not comparable to better than ~1 s.