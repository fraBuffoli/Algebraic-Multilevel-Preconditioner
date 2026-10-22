#!/bin/bash
#SBATCH --nodes=5                    	# 5 nodes
#SBATCH --ntasks-per-node=48         	# 48 tasks per node -> max 240 MPI processes
#SBATCH --exclusive                  	# whole nodes: clean timings
#SBATCH --partition=g100_usr_prod    	# server type
#SBATCH --error=allrun_p5.err         	# standard error file
#SBATCH --output=allrun_p5.out        	# standard output file
#SBATCH --account=IscrC_EXTA-PMF       	# account name
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=francesco.buffoli@mail.polimi.it
#SBATCH --job-name=s2l_p5
#SBATCH --time=01:00:00
# -------------------------------------------------------------------------------- #
# P5 - small coarse space (m_i, hence n0) and fast coarse solve. No code change: the
# campaign only varies existing options on the bubbles time loops of P1/P2.
#
#   SWEEP (--reuse auto, time loops): how many coarse vectors per subdomain are worth it?
#     ref    --tau 0.3  --nev 60   the setting of P1/P2 (m_i ~ 25 at 128^3, ~52 at 192^3)
#     t010   --tau 0.1  --nev 60   fewer vectors by a stricter threshold |lambda| >= 10
#     t003   --tau 0.03 --nev 60   |lambda| >= 33
#     n32/n16/n8   --tau 0.3 --nev 32/16/8   fewer vectors by a cap on m_i
#   n0 = sum m_i decides the coarse solver (root below 10000, as in P1/P2).
#
#   COARSE (--reuse refactor, a few steps): where to solve A_00 at the ref setting (n0 ~ 12600):
#     cs_root   A_00 gathered on rank 0
#     cs_cpK    cluster_sparse_solver on K leaders; a leader owns the rows of N/K consecutive
#               ranks, so K = number of nodes means one leader per node (gathers inside a node)
#
#   1. from test/p5:               sbatch Allrun_p5.sh
#      larger run (where P5 should matter most):  NP=960 sbatch --nodes=20 Allrun_p5.sh
#   2. summary + plot:             python3 report_p5.py run_p5_<jobid>
# The binary is built inside the run directory (does not touch ../../bin).
# Local smoke test of this script (small sizes, any MPI):
#   SMOKE=1 NP=4 ./Allrun_p5.sh
# -------------------------------------------------------------------------------- #

## what to run (1 = on), in this order                                     (estimates on 5 nodes)
RUN_CHECK=${RUN_CHECK:-0}        # unit tests (same code as P2: off by default)                 ~ 6 min
RUN_LARGE=${RUN_LARGE:-1}        # bubbles3d 192^3, rho 1000: ref t010 t003 n32 n16 n8 (30 steps) ~ 22 min
RUN_COARSE=${RUN_COARSE:-1}      # bubbles3d 192^3, ref: root, cpardiso on K leaders (4 steps)    ~ 5 min
RUN_SEQ=${RUN_SEQ:-1}            # bubbles3d 128^3, rho 1000: ref t010 n16 n8 (30 steps)           ~ 7 min

NP=${NP:-240}                    # processes = subdomains
NODES=${NODES:-${SLURM_JOB_NUM_NODES:-$(( (NP + 47) / 48 ))}}
RUN_TIMEOUT=${RUN_TIMEOUT:-600}  # seconds: a run that does not finish is stopped, the job goes on
COARSE=${COARSE:-"--coarse-solver cpardiso --coarse-procs 48 --coarse-root-below 10000"}
STEPS=${STEPS:-30}               # time steps of the sweep runs
STEPS_CS=${STEPS_CS:-4}          # time steps of the coarse-solver runs (refactor: setup + 3 refactorizations)
CORR=${CORR:-3}                  # correctors per step (PISO-like)
N_SEQ=${N_SEQ:-128}; N_LARGE=${N_LARGE:-192}
# Bubbles of P1/P2: radius 0.04 (5 cells at 128^3), 150 bubbles, rho 1000.
BUB="--generate bubbles3d --nbubbles 150 --radius 0.04 --cfl 0.5 --rho-ratio 1000"
SWEEP_LARGE=${SWEEP_LARGE:-"ref t010 t003 n32 n16 n8"}
SWEEP_SEQ=${SWEEP_SEQ:-"ref t010 n16 n8"}
CS_PROCS=${CS_PROCS:-"$NODES $((4 * NODES)) 48"}   # leaders of cluster_sparse_solver

# coarse-space options of a sweep variant
variant() {
    case $1 in
        ref)  echo "--tau 0.3 --nev 60" ;;
        t010) echo "--tau 0.1 --nev 60" ;;
        t003) echo "--tau 0.03 --nev 60" ;;
        n32)  echo "--tau 0.3 --nev 32" ;;
        n16)  echo "--tau 0.3 --nev 16" ;;
        n8)   echo "--tau 0.3 --nev 8" ;;
        n4)   echo "--tau 0.3 --nev 4" ;;
        *)    echo "unknown variant $1" >&2; exit 1 ;;
    esac
}

# -------------------------------------------------------------------------------- #
if [ "${SMOKE:-0}" = 1 ]; then        # local check of the script: tiny problems, no modules
    module() { :; }
    N_SEQ=24; N_LARGE=28; STEPS=4; STEPS_CS=2; NODES=1; CS_PROCS="1 2 48"
    BUB="--generate bubbles3d --nbubbles 20 --radius 0.1 --cfl 1 --rho-ratio 1000"
    COARSE="--coarse-solver cpardiso --coarse-procs 2 --coarse-root-below 40"
    MPIRUN_OPTS=${MPIRUN_OPTS:-"--oversubscribe"}; NCHECK=${NCHECK:-4}
fi
NCHECK=${NCHECK:-16}

module load profile/eng
module load autoload openfoam/10
module load metis/5.1.0--gcc--10.2.0
module load intel-oneapi-mkl/2023.2.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # one sequential factorization per process
unset I_MPI_PMI_LIBRARY                         # silences 240 "will be ignored" warnings per run

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
REPO=$(cd ../.. && pwd)
OUT=$PWD/run_p5_${SLURM_JOB_ID:-local}
BIN=$OUT/bin/schwarz2lvl
mkdir -p $OUT/csv
module list > $OUT/log.modules 2>&1
if git -C $REPO rev-parse --git-dir > /dev/null 2>&1; then
    git -C $REPO log -1 --format='%h %s' > $OUT/log.git 2>&1
    git -C $REPO status --short -- src include test > $OUT/git_status.txt 2>&1
else
    echo "not a git checkout: see src_hash.txt" > $OUT/log.git
fi
# Fingerprint of the sources actually compiled (works also outside a git checkout).
( cd $REPO && find src include test -type f \( -name '*.cpp' -o -name '*.hpp' \) | sort | xargs md5sum ) > $OUT/src_hash.txt
echo "sources md5: $(md5sum < $OUT/src_hash.txt | cut -c1-12)" >> $OUT/log.git
echo "NP=$NP NODES=$NODES COARSE='$COARSE' STEPS=$STEPS STEPS_CS=$STEPS_CS CORR=$CORR N_SEQ=$N_SEQ N_LARGE=$N_LARGE CS_PROCS='$CS_PROCS'" > $OUT/config.txt

# run NAME NP <schwarz2lvl options>  ->  log.NAME, csv/NAME.csv (one line per solve), status.txt
run() {
    local name=$1 np=$2 t0 st rc; t0=$(date +%s); shift 2
    printf "\n--- %-24s np=%-4s " "$name" "$np"
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
        ( cd $OUT && timeout $RUN_TIMEOUT mpirun $MPIRUN_OPTS -n $NCHECK $OUT/bin/$t > $OUT/log.$t 2>&1 )
        printf "\n    %-20s %s" $t "$(tail -n1 $OUT/log.$t)"
    done
fi

# LARGE - subdomains of ~30k cells at 240 processes (~7k at 960). At the ref setting m_i hits
# the cap nev = 60 on some subdomains and n0 ~ 12600 goes to cpardiso; the variants shrink m_i.
if [ $RUN_LARGE = 1 ]; then
    for v in $SWEEP_LARGE; do
        run bub${N_LARGE}_$v $NP $BUB --nx $N_LARGE --steps $STEPS --reuse auto $(variant $v) $COARSE
    done
fi

# COARSE - same matrix sequence, ref setting, only the place where A_00 is factorized/solved
# changes. --reuse refactor: no rebuild decisions, every run does the same operations.
if [ $RUN_COARSE = 1 ]; then
    run bub${N_LARGE}_cs_root $NP $BUB --nx $N_LARGE --steps $STEPS_CS --reuse refactor $(variant ref) \
        --coarse-solver root
    for k in $CS_PROCS; do
        run bub${N_LARGE}_cs_cp$k $NP $BUB --nx $N_LARGE --steps $STEPS_CS --reuse refactor $(variant ref) \
            --coarse-solver cpardiso --coarse-procs $k --coarse-root-below 0
    done
fi

# SEQ - 128^3 (~9k cells per subdomain): n0 ~ 6000 already on rank 0 at the ref setting.
if [ $RUN_SEQ = 1 ]; then
    for v in $SWEEP_SEQ; do
        run bub${N_SEQ}_$v $NP $BUB --nx $N_SEQ --steps $STEPS --reuse auto $(variant $v) $COARSE
    done
fi

printf "\n--- done: python3 report_p5.py %s\n" "$(basename $OUT)"
