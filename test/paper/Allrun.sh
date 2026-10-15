#!/bin/bash
#SBATCH --nodes=5                    	# 5 nodes
#SBATCH --ntasks-per-node=48         	# 48 tasks per node -> max 240 MPI processes
#SBATCH --exclusive                  	# whole nodes: clean timings
#SBATCH --partition=g100_usr_prod    	# server type
#SBATCH --error=allrun.err            	# standard error file
#SBATCH --output=allrun.out           	# standard output file
#SBATCH --account=IscrC_EXTA-PMF       	# account name
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=francesco.buffoli@mail.polimi.it
#SBATCH --job-name=s2l_paper
#SBATCH --time=05:00:00
# -------------------------------------------------------------------------------- #
# Reproduction of the experiments of Section 4 of
#   Al Daas, Jolivet, Rees, "Efficient algebraic two-level Schwarz preconditioner
#   for sparse matrices", SIAM J. Sci. Comput. 45 (2023)
# within 5 nodes (one subdomain per MPI process), with
#   - A_00 factorized in parallel: MKL cluster_sparse_solver on 48 processes (COARSE)
#     when n0 >= 10000, gathered on rank 0 below (--coarse-root-below 10000: crossover of
#     the root / cp48 calibration of run 22861093, Allrun_perf.sh);
#   - adaptive local eigensolver: k = 32, 64, ... eigenpairs until |lambda| < 1/tau
#     (default --nev-start 32), same selected vectors as computing nev = 300;
#   - PARDISO (local and cluster) without iterative refinement, kernel of A~_ii from
#     multi-RHS solves, timers of the preconditioner application (since run 22861093);
#   - Table 1 also at tau = 0.9 (the paper's G3_circuit row is its Fig. 2 at tau = 0.9).
#
#   1. once, on a LOGIN node (compute nodes are offline):  ./get_matrices.sh
#   2. from test/paper:                                     sbatch Allrun.sh
#   3. tables and plots:                                    python3 plot_paper.py run_<jobid>
#      against the previous run (same names):               python3 plot_paper.py run_<jobid> --ref run_22843541
# The binary is built inside run_<jobid>/: ../../bin is not touched.
# -------------------------------------------------------------------------------- #

## what to run (1 = on), in this order (cheapest first)
RUN_CHECK=1          # build + MPI unit tests on 16 processes                       ~ 5 min
RUN_FIG4=1           # strong scaling, 3D convection-diffusion, nu = 1e-2 (Fig. 4)  ~ 12 min
RUN_TAB2=1           # 2D/3D convection-diffusion, nu = 1 ... 1e-4 (Table 2)        ~ 5 min
RUN_TAB1=1           # SuiteSparse matrices, nev = 300, TAB1_TAUS (Table 1)         ~ 30-60 min
RUN_FIG2=1           # G3_circuit, sweep on tau (Fig. 2)                            ~ 5 min

NP=240               # processes = subdomains for Tables 1-2 and Fig. 2 (paper: 256)
RUN_TIMEOUT=1200     # seconds: a run that does not finish is stopped, the job goes on
COARSE="--coarse-solver cpardiso --coarse-procs 48 --coarse-root-below 10000"   # rank 0 only: --coarse-solver root
TAB1_TAUS="0.6 0.9"  # 0.6 = paper setting -> tab1_<m>_2L; others -> tab1_<m>_2L_t<tau>
MATRICES="finan512 nxp1 CoupCons3D ecology2 tmt_unsym thermal2 atmosmodj G3_circuit Transport memchip circuit5M_dc"
# circuit5M_dc: the two-level setup was still running after 20 min in run 22838900. --verbose 2 prints
# one line per rank when its eigensolver ends (grep "eigensolver done" log.tab1_circuit5M_dc_2L*).
EXTRA_circuit5M_dc="--verbose 2"

# -------------------------------------------------------------------------------- #
# environment: the OpenFOAM 10 one (compiler + Intel MPI) + METIS + MKL (PARDISO).
# The MKL module without MPI tag does not replace OpenFOAM's Intel MPI.
module load profile/eng
module load autoload openfoam/10
module load metis/5.1.0--gcc--10.2.0
module load intel-oneapi-mkl/2023.2.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # one sequential LU per process

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
REPO=$(cd ../.. && pwd)
OUT=$PWD/run_${SLURM_JOB_ID:-local}
BIN=$OUT/bin/schwarz2lvl
MAT=$REPO/matrices
mkdir -p $OUT/csv $OUT/hist
module list > $OUT/log.modules 2>&1
git -C $REPO log -1 --format='%h %s' > $OUT/log.git 2>&1
git -C $REPO status --short -- src include > $OUT/git_status.txt 2>&1   # uncommitted changes (not log.*: plot_paper.py reads log.* as runs)
echo "NP=$NP COARSE='$COARSE' TAB1_TAUS='$TAB1_TAUS'" > $OUT/config.txt

# -------------------------------------------------------------------------------- #
# run NAME NP <schwarz2lvl options>  ->  log.NAME, csv/NAME.csv, hist/NAME.txt, status.txt
run() {
    local name=$1 np=$2 t0 st rc; t0=$(date +%s); shift 2
    printf "\n--- %-28s np=%-4s " "$name" "$np"
    timeout $RUN_TIMEOUT mpirun -n $np $BIN "$@" \
        --csv $OUT/csv/$name.csv --history $OUT/hist/$name.txt > $OUT/log.$name 2>&1
    rc=$?
    # outcome read from the log: the exit code of mpirun is not reliable across MPI versions
    if   grep -q "NOT converged in" $OUT/log.$name; then st=not_converged
    elif grep -q " converged in"     $OUT/log.$name; then st=ok
    elif [ $rc = 124 ];                              then st=timeout
    else                                                  st=error; fi
    printf "%s (%ss)" $st $(( $(date +%s) - t0 ))
    echo "$name $np $st $(( $(date +%s) - t0 )) $rc" >> $OUT/status.txt
}
t1name() { [ $2 = 0.6 ] && echo tab1_$1_2L || echo tab1_$1_2L_t$2; }   # matrix tau -> run name

# -------------------------------------------------------------------------------- #
printf "\n--- Building (USE_CPARDISO=1) ---"
libdir() { local d; for d in ${LD_LIBRARY_PATH//:/ }; do ls $d/$1.so* > /dev/null 2>&1 && { echo $d; return; }; done; }
METIS_DIR=$(libdir libmetis)                          # .../metis-5.1.0-xxx/lib
MKL_DIR=${MKLROOT:+$MKLROOT/lib/intel64}; MKL_DIR=${MKL_DIR:-$(libdir libmkl_core)}   # .../mkl/2023.2.0/lib/intel64
ls $MKL_DIR/libmkl_blacs_intelmpi_lp64.so* > /dev/null 2>&1 ||
    { printf "\nMKL BLACS for Intel MPI not found in %s: no cluster_sparse_solver (use COARSE=--coarse-solver root)\n" "$MKL_DIR"; exit 1; }
( cd $REPO && make -j 48 all tests BUILD=$OUT/build BIN=$OUT/bin \
    CXX=mpicxx EIGEN_INC=external/eigen3 CXXFLAGS="-std=c++17 -O3 -march=skylake-avx512 -DNDEBUG" \
    METIS_INC="$(dirname "$METIS_DIR")/include" METIS_LIB="-L$METIS_DIR -Wl,-rpath,$METIS_DIR -lmetis" \
    USE_MUMPS=0 USE_UMFPACK=0 USE_PARDISO=1 USE_CPARDISO=1 MKL_INC="$(dirname "$(dirname "$MKL_DIR")")/include" \
    MKL_LIB="-L$MKL_DIR -Wl,-rpath,$MKL_DIR -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl" \
    MKL_BLACS_LIB="-lmkl_blacs_intelmpi_lp64" \
  ) > $OUT/log.build 2>&1 || { printf "\nbuild FAILED: see %s/log.build\n" $OUT; exit 1; }

if [ $RUN_CHECK = 1 ]; then
    printf "\n--- Unit tests ---"
    mpirun -n 16 $OUT/bin/test_subdomain      > $OUT/log.test_subdomain 2>&1
    mpirun -n 16 $OUT/bin/test_preconditioner > $OUT/log.test_preconditioner 2>&1
    printf "\n    test_subdomain: %s   test_preconditioner: %s" \
        "$(tail -n1 $OUT/log.test_subdomain)" "$(tail -n1 $OUT/log.test_preconditioner)"
fi

# Fig. 4 - strong scaling of (4.1) in 3D, nu = 1e-2, tau = 0.3, nev = 60.
# Paper: n = 8.1e6 (P2), N = 32 ... 512. Here: n = 200^3 = 8.0e6 (finite volumes), N = 30 ... 240.
if [ $RUN_FIG4 = 1 ]; then
    for N in 30 60 120 240; do
        run fig4_N$N $N --generate convdiff3d --nx 200 --nu 1e-2 --rhs generated --tau 0.3 --nev 60 $COARSE
    done
fi

# Table 2 - nu = 1 ... 1e-4, tau = 0.3, nev = 60, with the subdomain size of the paper
# (~6k unknowns per subdomain in 2D, ~2k in 3D): 1200^2 and 78^3 unknowns on NP subdomains.
if [ $RUN_TAB2 = 1 ]; then
    for nu in 1 1e-1 1e-2 1e-3 1e-4; do
        run tab2_2d_nu${nu}_2L  $NP --generate convdiff2d --nx 1200 --nu $nu --rhs generated --tau 0.3 --nev 60 $COARSE
        run tab2_2d_nu${nu}_RAS $NP --generate convdiff2d --nx 1200 --nu $nu --rhs generated --coarse none
        run tab2_3d_nu${nu}_2L  $NP --generate convdiff3d --nx 78   --nu $nu --rhs generated --tau 0.3 --nev 60 $COARSE
        run tab2_3d_nu${nu}_RAS $NP --generate convdiff3d --nx 78   --nu $nu --rhs generated --coarse none
    done
fi

# Table 1 - RAS + deflated coarse correction (2.4), nev = 300, random rhs, GMRES(30), rtol 1e-8
# (the paper reports > 100 iterations as a failure). RAS = one-level reference.
if [ $RUN_TAB1 = 1 ]; then
    for m in $MATRICES; do
        [ -f $MAT/$m.mtx ] || { printf "\n--- %s: missing, run get_matrices.sh" $m; continue; }
        extra=EXTRA_$m
        run tab1_${m}_RAS $NP --matrix $MAT/$m.mtx --coarse none --maxit 200
        for tau in $TAB1_TAUS; do
            run $(t1name $m $tau) $NP --matrix $MAT/$m.mtx --tau $tau --nev 300 --maxit 200 $COARSE ${!extra}
            # a larger tau only makes the setup longer: after a timeout skip the other values
            [ "$(tail -n1 $OUT/status.txt | cut -d' ' -f3)" = timeout ] && break
        done
    done
fi

# Fig. 2 - G3_circuit, tau = 0.02 ... 0.9 (nev = 300), and the one-level ASM of the figure.
# The tau of TAB1_TAUS are not run again: plot_paper.py takes them from Table 1.
if [ $RUN_FIG2 = 1 ] && [ -f $MAT/G3_circuit.mtx ]; then
    run fig2_ASM $NP --matrix $MAT/G3_circuit.mtx --one-level asm --coarse none --maxit 200
    for tau in 0.02 0.05 0.1 0.3 0.6 0.9; do
        [ $RUN_TAB1 = 1 ] && [[ " $TAB1_TAUS " == *" $tau "* ]] && continue
        run fig2_tau$tau $NP --matrix $MAT/G3_circuit.mtx --tau $tau --nev 300 --maxit 200 $COARSE
    done
fi

printf "\n--- Done: results in %s\n" $OUT
