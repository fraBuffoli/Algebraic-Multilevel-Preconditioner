#!/bin/bash
#SBATCH --nodes=5                    	# 5 nodes
#SBATCH --ntasks-per-node=48         	# 48 tasks per node -> max 240 MPI processes
#SBATCH --exclusive                  	# whole nodes: clean timings
#SBATCH --partition=g100_usr_prod    	# server type
#SBATCH --error=allrun_reuse.err      	# standard error file
#SBATCH --output=allrun_reuse.out     	# standard output file
#SBATCH --account=IscrC_EXTA-PMF       	# account name
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=francesco.buffoli@mail.polimi.it
#SBATCH --job-name=s2l_reuse
#SBATCH --time=01:00:00
# -------------------------------------------------------------------------------- #
# P1 - setup reuse in a sequence of solves with the same sparsity pattern (CFD time loop).
# Every run is a time loop of STEPS steps x CORR correctors (same matrix, new rhs):
#   --reuse off       full setup before every solve (no persistent state: the reference)
#   --reuse refactor  numerical refactorization only, the coarse basis is never rebuilt
#   --reuse auto      refactorize, rebuild at break-even (time lost to the old basis = rebuild premium)
#
#   1. from test/reuse:            sbatch Allrun_reuse.sh
#   2. summary + plots:            python3 plot_reuse.py run_reuse_<jobid>
# The binary is built inside the run directory (does not touch ../../bin).
# Local smoke test of this script (small sizes, any MPI):
#   SMOKE=1 NP=4 ./Allrun_reuse.sh
# -------------------------------------------------------------------------------- #

## what to run (1 = on), in this order                                     (estimates on 5 nodes)
RUN_CHECK=${RUN_CHECK:-1}        # build + unit tests on 16 processes (incl. test_reuse)          ~ 6 min
RUN_SEQ=${RUN_SEQ:-1}            # bubbles3d 128^3 (~9k cells/process), rho 1000: off/refactor/auto ~ 8 min
RUN_CONTRAST=${RUN_CONTRAST:-1}  # same at rho 10 (refactor, auto) + one-level RAS at rho 1000     ~ 6 min
RUN_LARGE=${RUN_LARGE:-1}        # bubbles3d 192^3 (~30k cells/process): off / auto                ~ 15 min
RUN_NONSYM=${RUN_NONSYM:-1}      # convdiff3d 100^3, pulsating convection (nonsymmetric): off/auto ~ 5 min

NP=${NP:-240}                    # processes = subdomains
RUN_TIMEOUT=${RUN_TIMEOUT:-1500} # seconds: a run that does not finish is stopped, the job goes on
COARSE=${COARSE:-"--coarse-solver cpardiso --coarse-procs 48 --coarse-root-below 10000"}
STEPS=${STEPS:-60}               # time steps of the refactor/auto runs
STEPS_OFF=${STEPS_OFF:-15}       # off: every solve costs a full setup, a few steps give its mean cost
STEPS_LARGE=${STEPS_LARGE:-30}   # 192^3 runs
STEPS_LARGE_OFF=${STEPS_LARGE_OFF:-5}
CORR=${CORR:-3}                  # correctors per step (PISO-like)
N_SEQ=${N_SEQ:-128}; N_LARGE=${N_LARGE:-192}; N_NONSYM=${N_NONSYM:-100}
# Bubbles: radius 0.04 = 5 cells at 128^3, ~4% gas; subdomains of ~20 cells: bubbles cross the interfaces.
BUB="--generate bubbles3d --nbubbles 150 --radius 0.04 --cfl 0.5 --tau 0.3 --nev 60"
CD="--generate convdiff3d --nx $N_NONSYM --nu 1e-2 --pulse 0.3 --pulse-period 20 --tau 0.3 --nev 60"

# -------------------------------------------------------------------------------- #
if [ "${SMOKE:-0}" = 1 ]; then        # local check of the script: tiny problems, no modules
    module() { :; }
    N_SEQ=24; N_LARGE=28; N_NONSYM=20; STEPS=6; STEPS_OFF=2; STEPS_LARGE=4; STEPS_LARGE_OFF=2
    BUB="--generate bubbles3d --nbubbles 20 --radius 0.1 --cfl 1 --tau 0.3 --nev 20"
    CD="--generate convdiff3d --nx $N_NONSYM --nu 1e-2 --pulse 0.3 --pulse-period 20 --tau 0.3 --nev 20"
    MPIRUN_OPTS=${MPIRUN_OPTS:-"--oversubscribe"}; NCHECK=${NCHECK:-4}
fi
NCHECK=${NCHECK:-16}

module load profile/eng
module load autoload openfoam/10
module load metis/5.1.0--gcc--10.2.0
module load intel-oneapi-mkl/2023.2.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # one sequential LU per process

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
REPO=$(cd ../.. && pwd)
OUT=$PWD/run_reuse_${SLURM_JOB_ID:-local}
BIN=$OUT/bin/schwarz2lvl
mkdir -p $OUT/csv
module list > $OUT/log.modules 2>&1
git -C $REPO log -1 --format='%h %s' > $OUT/log.git 2>&1
git -C $REPO status --short -- src include test > $OUT/git_status.txt 2>&1
echo "NP=$NP COARSE='$COARSE' STEPS=$STEPS STEPS_OFF=$STEPS_OFF CORR=$CORR N_SEQ=$N_SEQ N_LARGE=$N_LARGE" > $OUT/config.txt

# run NAME NP <schwarz2lvl options>  ->  log.NAME, csv/NAME.csv (one line per solve), status.txt
run() {
    local name=$1 np=$2 t0 st rc; t0=$(date +%s); shift 2
    printf "\n--- %-28s np=%-4s " "$name" "$np"
    timeout $RUN_TIMEOUT mpirun $MPIRUN_OPTS -n $np $BIN "$@" --correctors $CORR \
        --steps-csv $OUT/csv/$name.csv > $OUT/log.$name 2>&1
    rc=$?
    if   grep -q "NOT converged" $OUT/log.$name;          then st=not_converged
    elif grep -q "^Time loop summary" $OUT/log.$name;     then st=ok
    elif [ $rc = 124 ];                                   then st=timeout
    else                                                       st=error; fi
    printf "%s (%ss)" $st $(( $(date +%s) - t0 ))
    echo "$name $np $st $(( $(date +%s) - t0 )) $rc" >> $OUT/status.txt
}

# -------------------------------------------------------------------------------- #
printf "\n--- Building ---"
if [ "${SMOKE:-0}" = 1 ]; then
    ( cd $REPO && make -j 4 all tests BUILD=$OUT/build BIN=$OUT/bin ) > $OUT/log.build 2>&1 ||
        { printf "\nbuild FAILED: see %s/log.build\n" $OUT; exit 1; }
else
    libdir() { local d; for d in ${LD_LIBRARY_PATH//:/ }; do ls $d/$1.so* > /dev/null 2>&1 && { echo $d; return; }; done; }
    METIS_DIR=$(libdir libmetis)
    MKL_DIR=${MKLROOT:+$MKLROOT/lib/intel64}; MKL_DIR=${MKL_DIR:-$(libdir libmkl_core)}
    ls $MKL_DIR/libmkl_blacs_intelmpi_lp64.so* > /dev/null 2>&1 ||
        { printf "\nMKL BLACS for Intel MPI not found in %s\n" "$MKL_DIR"; exit 1; }
    ( cd $REPO && make -j 48 all tests BUILD=$OUT/build BIN=$OUT/bin \
        CXX=mpicxx EIGEN_INC=external/eigen3 CXXFLAGS="-std=c++17 -O3 -march=skylake-avx512 -DNDEBUG" \
        METIS_INC="$(dirname "$METIS_DIR")/include" METIS_LIB="-L$METIS_DIR -Wl,-rpath,$METIS_DIR -lmetis" \
        USE_MUMPS=0 USE_UMFPACK=0 USE_PARDISO=1 USE_CPARDISO=1 MKL_INC="$(dirname "$(dirname "$MKL_DIR")")/include" \
        MKL_LIB="-L$MKL_DIR -Wl,-rpath,$MKL_DIR -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl" \
        MKL_BLACS_LIB="-lmkl_blacs_intelmpi_lp64" \
      ) > $OUT/log.build 2>&1 || { printf "\nbuild FAILED: see %s/log.build\n" $OUT; exit 1; }
fi

if [ $RUN_CHECK = 1 ]; then
    printf "\n--- Unit tests (%s processes) ---" $NCHECK
    for t in test_subdomain test_preconditioner test_reuse; do
        ( cd $OUT && mpirun $MPIRUN_OPTS -n $NCHECK $OUT/bin/$t > $OUT/log.$t 2>&1 )   # test_reuse writes files in cwd
        printf "\n    %-20s %s" $t "$(tail -n1 $OUT/log.$t)"
    done
    grep -q "ALL PASSED" $OUT/log.test_reuse || { printf "\ntest_reuse FAILED: the sequence runs are skipped\n"; exit 1; }
fi

# SEQ - the main comparison: same problem, three reuse modes. off runs fewer steps (its cost per
# solve is stationary); plot_reuse.py compares the mean cost per solve.
if [ $RUN_SEQ = 1 ]; then
    run bub${N_SEQ}_r1000_off      $NP $BUB --nx $N_SEQ --rho-ratio 1000 --steps $STEPS_OFF --reuse off      $COARSE
    run bub${N_SEQ}_r1000_refactor $NP $BUB --nx $N_SEQ --rho-ratio 1000 --steps $STEPS     --reuse refactor $COARSE
    run bub${N_SEQ}_r1000_auto     $NP $BUB --nx $N_SEQ --rho-ratio 1000 --steps $STEPS     --reuse auto     $COARSE
fi

# CONTRAST - how fast the coarse basis ages depends on the contrast; one-level RAS shows what the
# coarse space is worth at rho 1000 (with the same reuse of its LU).
if [ $RUN_CONTRAST = 1 ]; then
    run bub${N_SEQ}_r10_refactor   $NP $BUB --nx $N_SEQ --rho-ratio 10   --steps $STEPS --reuse refactor $COARSE
    run bub${N_SEQ}_r10_auto       $NP $BUB --nx $N_SEQ --rho-ratio 10   --steps $STEPS --reuse auto     $COARSE
    run bub${N_SEQ}_r1000_ras      $NP $BUB --nx $N_SEQ --rho-ratio 1000 --steps $STEPS --reuse auto --coarse none --maxit 2000
fi

# LARGE - subdomains of ~30k cells (as fig4 at N = 240, setup ~16 s): where the full setup at
# every solve hurts most.
if [ $RUN_LARGE = 1 ]; then
    run bub${N_LARGE}_r1000_off    $NP $BUB --nx $N_LARGE --rho-ratio 1000 --steps $STEPS_LARGE_OFF --reuse off  $COARSE
    run bub${N_LARGE}_r1000_auto   $NP $BUB --nx $N_LARGE --rho-ratio 1000 --steps $STEPS_LARGE     --reuse auto $COARSE
fi

# NONSYM - nonsymmetric LU: numerical refactorization with the scaling/matching of the first
# analysis (PARDISO falls back to a full factorization if more pivots are perturbed).
if [ $RUN_NONSYM = 1 ]; then
    run cd${N_NONSYM}_pulse_off    $NP $CD --rhs random --steps $STEPS_OFF --reuse off  $COARSE
    run cd${N_NONSYM}_pulse_auto   $NP $CD --rhs random --steps $STEPS     --reuse auto $COARSE
fi

printf "\n--- done: python3 plot_reuse.py %s\n" "$(basename $OUT)"
