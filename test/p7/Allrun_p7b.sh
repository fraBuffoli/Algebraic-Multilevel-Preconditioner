#!/bin/bash
#SBATCH --nodes=5                    	# 5 nodes
#SBATCH --ntasks-per-node=48         	# 48 tasks per node -> 240 MPI processes
#SBATCH --exclusive                  	# whole nodes: clean timings
#SBATCH --partition=g100_usr_prod    	# server type
#SBATCH --error=allrun_p7b.err        	# standard error file
#SBATCH --output=allrun_p7b.out       	# standard output file
#SBATCH --account=IscrC_EXTA-PMF       	# account name
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=francesco.buffoli@mail.polimi.it
#SBATCH --job-name=s2l_p7b
#SBATCH --time=01:00:00
# -------------------------------------------------------------------------------- #
# P7b - second campaign of the inexact local solvers, after the corrections suggested by the
# first one (run_p7_23018578 at 240 processes, run_p7_23019000 at 960):
#   * only AMG with Gauss-Seidel is left: coarsening sa (default) or rs (IC, spai0, Chebyshev removed);
#   * coarsest AMG level <= 500 unknowns (--amg-coarse-enough, AMGCL default 3000 made the setup of
#     17k-cell subdomains more expensive than an exact Cholesky factorization);
#   * full AMG setup at every refactorization (--amg-reuse 0: the reused hierarchy aged with the bubbles);
#   * reuse policy with robust costs (lower median of the measured times) and --rebuild-growth 4.
# Same problems as the first campaign, fewer steps and variants (~30% of its cost):
#   bub192: 192^3, ~30k cells per process at 240 processes, 20 steps
#   bub256: 256^3, ~70k cells per process at 240 processes, 16 steps
# The 960-process version is Allrun_p7b_960.sh (it calls this script with other settings).
#
#   1. from test/p7:               sbatch Allrun_p7b.sh                 (~35 min, ~140 core-hours)
#   2. summary + plots:            python3 report_p7.py run_p7b_<jobid>
# Local smoke test of this script (small sizes, any MPI):
#   SMOKE=1 NP=4 ./Allrun_p7b.sh
# -------------------------------------------------------------------------------- #

## what to run (1 = on), in this order                                    (estimates on 5 nodes)
RUN_CHECK=${RUN_CHECK:-1}        # unit tests test_inexact + test_reuse (new code)                ~ 2 min
RUN_A=${RUN_A:-1}                # bubbles3d 192^3, 5 variants, 20 steps                          ~ 9 min
RUN_B=${RUN_B:-1}                # bubbles3d 256^3, 6 variants, 16 steps                          ~ 21 min

NP=${NP:-${SLURM_NTASKS:-240}}   # processes = subdomains
RUN_TIMEOUT=${RUN_TIMEOUT:-900}  # seconds: a run that does not finish is stopped, the job goes on
COARSE=${COARSE:-"--coarse-solver cpardiso --coarse-procs 48 --coarse-root-below 10000"}
STEPS_A=${STEPS_A:-20}           # time steps, 192^3
STEPS_B=${STEPS_B:-16}           # time steps, 256^3 (rank 0 generates and sends 16.8M values per step)
CORR=${CORR:-3}                  # correctors per step (PISO-like)
N_A=${N_A:-192}; N_B=${N_B:-256}
BUB="--generate bubbles3d --nbubbles 150 --radius 0.04 --cfl 0.5 --rho-ratio 1000 --reuse auto"
BASE=${BASE:-"--tau 0.3 --nev 8 --eig-tol 1e-4"}   # best coarse space of P5, common to all variants
# ref = exact Cholesky (PARDISO); sp = exact, single precision; sa / rs = AMG V-cycle with Gauss-Seidel,
# smoothed aggregation / Ruge-Stuben (new defaults: coarsest <= 500, full setup at every step);
# sasp = sa in float; sare = sa with the hierarchy reused in the refactorizations (old default).
VARIANTS_A=${VARIANTS_A:-"ref sa sp rs sasp"}
VARIANTS_B=${VARIANTS_B:-"ref sa sp rs sasp sare"}
OUT_PREFIX=${OUT_PREFIX:-run_p7b}

# local-solver options of a variant (added to BASE)
variant() {
    local amg="--local-approx amg"
    case $1 in
        ref)     echo "" ;;
        sp)      echo "--local-precision single" ;;
        sa)      echo "$amg --amg-coarsening sa" ;;
        rs)      echo "$amg --amg-coarsening rs" ;;
        sasp)    echo "$amg --amg-coarsening sa --local-precision single" ;;
        sare)    echo "$amg --amg-coarsening sa --amg-reuse 1" ;;
        sa2)     echo "$amg --amg-coarsening sa --amg-sweeps 2" ;;
        *)       echo "unknown variant $1" >&2; exit 1 ;;
    esac
}

# -------------------------------------------------------------------------------- #
if [ "${SMOKE:-0}" = 1 ]; then        # local check of the script: tiny problems, no modules
    module() { :; }
    N_A=24; N_B=32; STEPS_A=3; STEPS_B=2
    BUB="--generate bubbles3d --nbubbles 20 --radius 0.1 --cfl 1 --rho-ratio 1000 --reuse auto"
    COARSE="--coarse-solver root"
    MPIRUN_OPTS=${MPIRUN_OPTS:-"--oversubscribe"}; NCHECK=${NCHECK:-4}
fi
NCHECK=${NCHECK:-16}

module load profile/eng
module load autoload openfoam/10
module load metis/5.1.0--gcc--10.2.0
module load intel-oneapi-mkl/2023.2.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # one sequential factorization / V-cycle per process
unset I_MPI_PMI_LIBRARY                         # silences the "will be ignored" warnings of every rank

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
REPO=$(cd ../.. && pwd)
OUT=$PWD/${OUT_PREFIX}_${SLURM_JOB_ID:-local}
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
echo "NP=$NP COARSE='$COARSE' BASE='$BASE' STEPS_A=$STEPS_A STEPS_B=$STEPS_B CORR=$CORR N_A=$N_A N_B=$N_B" > $OUT/config.txt
# read by report_p7.py: "<variant> <options>" in the order of the campaign
for v in $(echo $VARIANTS_A $VARIANTS_B | tr ' ' '\n' | awk '!seen[$0]++'); do
    echo "$v $BASE $(variant $v)"
done > $OUT/variants.txt

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
TARGETS="all"; [ $RUN_CHECK = 1 ] && TARGETS="all tests"   # no unit tests: skip building them
if [ "${SMOKE:-0}" = 1 ]; then
    ( cd $REPO && make -j 4 $TARGETS BUILD=$OUT/build BIN=$OUT/bin USE_AMGCL=1 ) > $OUT/log.build 2>&1 ||
        { printf "\nbuild FAILED: see %s/log.build\n" $OUT; exit 1; }
else
    libdir() { local d; for d in ${LD_LIBRARY_PATH//:/ }; do ls $d/$1.so* > /dev/null 2>&1 && { echo $d; return; }; done; }
    METIS_DIR=$(libdir libmetis)
    MKL_DIR=${MKLROOT:+$MKLROOT/lib/intel64}; MKL_DIR=${MKL_DIR:-$(libdir libmkl_core)}
    ls $MKL_DIR/libmkl_blacs_intelmpi_lp64.so* > /dev/null 2>&1 ||
        { printf "\nMKL BLACS for Intel MPI not found in %s\n" "$MKL_DIR"; exit 1; }
    ( cd $REPO && make -j 48 $TARGETS BUILD=$OUT/build BIN=$OUT/bin \
        CXX=mpicxx EIGEN_INC=external/eigen3 CXXFLAGS="-std=c++17 -O3 -march=skylake-avx512 -DNDEBUG" \
        METIS_INC="$(dirname "$METIS_DIR")/include" METIS_LIB="-L$METIS_DIR -Wl,-rpath,$METIS_DIR -lmetis" \
        USE_MUMPS=0 USE_UMFPACK=0 USE_PARDISO=1 USE_CPARDISO=1 USE_AMGCL=1 \
        MKL_INC="$(dirname "$(dirname "$MKL_DIR")")/include" \
        MKL_LIB="-L$MKL_DIR -Wl,-rpath,$MKL_DIR -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl" \
        MKL_BLACS_LIB="-lmkl_blacs_intelmpi_lp64" \
      ) > $OUT/log.build 2>&1 || { printf "\nbuild FAILED: see %s/log.build\n" $OUT; exit 1; }
fi
[ -f $REPO/external/amgcl/amgcl/amg.hpp ] && grep -q -- "-DUSE_AMGCL" $OUT/log.build ||
    { printf "\nthis build has no AMGCL (external/amgcl missing?): P7 is required\n"; exit 1; }
timeout 120 mpirun $MPIRUN_OPTS -n 1 $BIN --help 2>/dev/null | grep -q -- "--amg-coarse-enough" ||
    { printf "\nthis build has no --amg-coarse-enough: the P7b sources are required\n"; exit 1; }

if [ $RUN_CHECK = 1 ]; then
    printf "\n--- Unit tests (%s processes) ---" $NCHECK
    for t in test_inexact test_reuse; do
        ( cd $OUT && timeout $RUN_TIMEOUT mpirun $MPIRUN_OPTS -n $NCHECK $OUT/bin/$t > $OUT/log.$t 2>&1 )
        printf "\n    %-20s %s" $t "$(tail -n1 $OUT/log.$t)"
    done
fi

# A - 192^3 (~30k cells per process at 240 processes).
if [ $RUN_A = 1 ]; then
    for v in $VARIANTS_A; do
        run bub${N_A}_$v $NP $BUB --nx $N_A --steps $STEPS_A $BASE $(variant $v) $COARSE
    done
fi

# B - 256^3 (~70k cells per process at 240 processes).
if [ $RUN_B = 1 ]; then
    for v in $VARIANTS_B; do
        run bub${N_B}_$v $NP $BUB --nx $N_B --steps $STEPS_B $BASE $(variant $v) $COARSE
    done
fi

printf "\n--- done: python3 report_p7.py %s\n" "$(basename $OUT)"
