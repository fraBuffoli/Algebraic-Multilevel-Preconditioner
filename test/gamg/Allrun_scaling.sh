#!/bin/bash
#SBATCH --nodes=5                    	# 5 nodes = 240 processes (submit_scaling.sh: --nodes 5, 7, 9)
#SBATCH --ntasks-per-node=48         	# 48 tasks per node
#SBATCH --exclusive                  	# whole nodes: clean timings
#SBATCH --partition=g100_usr_prod    	# server type
#SBATCH --error=allrun_scaling.err   	# standard error file
#SBATCH --output=allrun_scaling.out  	# standard output file
#SBATCH --account=IscrC_EXTA-PMF       	# account name
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=francesco.buffoli@mail.polimi.it
#SBATCH --job-name=s2l_scaling
#SBATCH --time=00:50:00
# -------------------------------------------------------------------------------- #
# Scaling campaign: GAMG of OpenFOAM (DIC smoother, the fastest OpenFOAM variant of the GAMG
# campaigns) against the two-level Schwarz preconditioner with AMG local solves (P7b setting),
# on the same pressure matrices (bubbles3d: 150 bubbles, rho ratio 1000, rising 0.5 cells per step),
# 192^3, 256^3 and 320^3, on the processes of the job (one job per process count: 240, 336, 432).
#
# Stopping test of both codes: the normalised residual of OpenFOAM, relTol RTOL = 1e-2 (the usual
# pressure tolerance of a time step). Every solve starts from 0, so the test is
#   sum_i |b - A x|_i <= 1e-2 sum_i |b|_i
# for both: OpenFOAM with relTol 1e-2 (tolerance 0), schwarz2lvl with --stop-norm openfoam --rtol 1e-2.
#
# Times of every solve, each between two synchronizations of all processes (csv/<run>.csv):
#   schwarz2lvl  t_update (initial setup / rebuild / refactorization / reuse) + t_solve (GMRES);
#                phases inside them in the timer table at the end of log.<run>
#   OpenFOAM     t_setup (construction of the solver: GAMG coarse matrices; agglomeration in the
#                first solve) + t_cycles (V-cycles); t_assembly (matrix assembly, not compared)
# Before the runs: one small schwarz2lvl run (log.warmup), not measured, and every schwarz2lvl run
# warms up its direct solver before the time loop (first factorization of a run took 13-17 s on
# G100 and inflated the initial setup and the rebuild cost of the reuse policy).
#
#   1. from test/gamg:   ./submit_scaling.sh      (Allprep_scaling.sh + this script on 5, 7 and 9 nodes)
#   2. plots + report:   python3 report_scaling.py run_scaling_np*_<jobid>   (or: run_scaling_*)
# Local smoke test (any MPI, OpenFOAM-10 environment sourced):  SMOKE=1 NP=4 ./Allrun_scaling.sh
# -------------------------------------------------------------------------------- #

NP=${NP:-${SLURM_NTASKS:-240}}   # processes = subdomains
SIZES=${SIZES:-"192 256 320"}    # cells per direction
STEPS=${STEPS:-16}               # time steps (same for all sizes: same amortization of the setup)
CORR=${CORR:-3}                  # solves per step (PISO-like correctors)
RTOL=${RTOL:-1e-2}               # relTol of the normalised residual of OpenFOAM, both codes
OF_SOLVER=${OF_SOLVER:-GAMG_DIC} # OpenFOAM variant (case/system/fvSolution)
NBUB=${NBUB:-150}; RADIUS=${RADIUS:-0.04}; RHO=${RHO:-1000}; CFL=${CFL:-0.5}; SEED=2024   # bubbles
BASE=${BASE:-"--tau 0.3 --nev 8 --eig-tol 1e-4 --local-approx amg"}   # P7b: AMG sa + Gauss-Seidel local solves
COARSE=${COARSE:-"--coarse-solver cpardiso --coarse-procs 48 --coarse-root-below 10000"}
STOP="--stop-norm openfoam --rtol $RTOL"
RUN_TIMEOUT=${RUN_TIMEOUT:-1200} # seconds: a run that does not finish is stopped, the job goes on
WARMUP_N=${WARMUP_N:-96}
OUT_PREFIX=${OUT_PREFIX:-run_scaling}

# -------------------------------------------------------------------------------- #
S2L_CXX=${S2L_CXX:-mpicxx}
S2L_CXXFLAGS=${S2L_CXXFLAGS:-"-std=c++17 -O3 -march=skylake-avx512 -DNDEBUG"}   # as in Allprep_scaling.sh
if [ "${SMOKE:-0}" = 1 ]; then        # local check of the script: tiny problems, no modules
    module() { :; }
    S2L_CXXFLAGS="-std=c++17 -O2 -DNDEBUG"
    SIZES=${SMOKE_SIZES:-"16 20"}; STEPS=3; WARMUP_N=12
    NBUB=20; RADIUS=0.1; CFL=1
    COARSE="--coarse-solver root"
    MPIRUN_OPTS=${MPIRUN_OPTS:-"--oversubscribe"}
    CASES=${CASES:-$PWD/cases_smoke}
fi

module load profile/eng
module load autoload openfoam/10
module load metis/5.1.0--gcc--10.2.0
module load intel-oneapi-mkl/2023.2.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # one sequential factorization / V-cycle per process
unset I_MPI_PMI_LIBRARY                         # silences the "will be ignored" warnings of every rank
[ -n "$WM_PROJECT_DIR" ] || { echo "OpenFOAM environment not loaded"; exit 1; }

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
REPO=$(cd ../.. && pwd)
OUT=$PWD/${OUT_PREFIX}_np${NP}_${SLURM_JOB_ID:-local}
CASES=${CASES:-$PWD/cases}
mkdir -p $OUT/csv $CASES
. ./gamg_functions.sh
module list > $OUT/log.modules 2>&1
if git -C $REPO rev-parse --git-dir > /dev/null 2>&1; then
    git -C $REPO log -1 --format='%h %s' > $OUT/log.git 2>&1
    git -C $REPO status --short -- src include test openfoam > $OUT/git_status.txt 2>&1
else
    echo "not a git checkout" > $OUT/log.git
fi
SRC=$(src_md5)
echo "sources md5: $SRC" >> $OUT/log.git
echo "NP=$NP SIZES='$SIZES' STEPS=$STEPS CORR=$CORR RTOL=$RTOL OF_SOLVER=$OF_SOLVER BASE='$BASE' COARSE='$COARSE' NBUB=$NBUB RADIUS=$RADIUS RHO=$RHO CFL=$CFL SEED=$SEED OpenFOAM=$WM_PROJECT_VERSION nodes=${SLURM_JOB_NUM_NODES:-1} job=${SLURM_JOB_ID:-local}" > $OUT/config.txt
# read by report_scaling.py: "<run> <kind> <options>"
{ echo "amg schwarz2lvl $BASE $STOP"; echo "$OF_SOLVER openfoam -solver $OF_SOLVER relTol $RTOL"; } > $OUT/variants.txt

# status line: NAME NP STATUS SECONDS RC
status() { printf "%s (%ss)" $3 $4; echo "$1 $2 $3 $4 $5" >> $OUT/status.txt; }

# run_of NAME CASE  ->  log.NAME, csv/NAME.csv
run_of() {
    local name=$1 dir=$2 t0 st rc; t0=$(date +%s)
    printf "\n--- %-24s np=%-4s " "$name" "$NP"
    local par=""; [ "$NP" -gt 1 ] && par="-parallel"
    timeout $RUN_TIMEOUT mpirun $MPIRUN_OPTS -n $NP $APP -case $dir $par -solver $OF_SOLVER -steps $STEPS \
        -correctors $CORR -csv $OUT/csv/$name.csv > $OUT/log.$name 2>&1
    rc=$?
    if   [ $rc = 124 ];                                                then st=timeout
    elif ! grep -q "^End" $OUT/log.$name;                              then st=error
    elif awk -F, 'NR > 1 && $11 == 0 { bad = 1 } END { exit !bad }' $OUT/csv/$name.csv; then st=not_converged
    else                                                                    st=ok; fi
    status $name $NP $st $(( $(date +%s) - t0 )) $rc
}

# run_s2l NAME N STEPS CORR LOG CSV <options>
s2l() {
    local n=$1 steps=$2 corr=$3 log=$4 csv=$5; shift 5
    timeout $RUN_TIMEOUT mpirun $MPIRUN_OPTS -n $NP $BIN/schwarz2lvl --generate bubbles3d --nx $n --steps $steps \
        --correctors $corr --nbubbles $NBUB --radius $RADIUS --rho-ratio $RHO --cfl $CFL --rhs generated \
        --reuse auto "$@" ${csv:+--steps-csv $csv} > $log 2>&1
}
run_s2l() {
    local name=$1 n=$2 t0 st rc; t0=$(date +%s)
    printf "\n--- %-24s np=%-4s " "$name" "$NP"
    s2l $n $STEPS $CORR $OUT/log.$name $OUT/csv/$name.csv $BASE $COARSE $STOP
    rc=$?
    if   grep -q "NOT converged" $OUT/log.$name;          then st=not_converged
    elif grep -q "^Time loop summary" $OUT/log.$name;     then st=ok
    elif [ $rc = 124 ];                                   then st=timeout
    else                                                       st=error; fi
    status $name $NP $st $(( $(date +%s) - t0 )) $rc
}

# -------------------------------------------------------------------------------- #
# Binaries: those of Allprep_scaling.sh if built from the same sources and checked, otherwise built
# and checked here (~5 minutes).
B=$CASES/bin_$SRC
if [ -f $B/OK ]; then
    printf "\n--- Binaries of Allprep_scaling.sh: %s (%s)" "$B" "$(cat $B/OK)"
    cp $B/OK $OUT/binaries.txt
else
    printf "\n--- Building and checking here (no checked binaries for sources %s in %s) " $SRC "$CASES"
    B=$OUT
    build_and_check $B > $OUT/log.build_and_check 2>&1 ||
        { printf "\nSTOP: %s\n" "$(cat $OUT/log.build_and_check)"; exit 1; }
    printf "%s" "$(cat $B/OK)"
    cp $B/OK $OUT/binaries.txt
fi
BIN=$B/bin
APP=$B/bin/bubblyPressureFoam

# Cases: prepared by Allprep_scaling.sh; a missing one is prepared here (idle cores, ~5-15 minutes).
for n in $SIZES; do
    d=$CASES/bub${n}_np$NP
    if ! case_ok $d $n $NP; then
        printf "\n--- preparing %s here (Allprep_scaling.sh did not prepare it: the other cores are idle) " "$d"
        prep_case $n $NP $d ${METHOD:-scotch} > $OUT/log.prep_$n 2>&1 || printf "FAILED: %s" $OUT/log.prep_$n
    fi
    case_ok $d $n $NP && cat $d/PREPARED >> $OUT/cases.txt
done

# Warm-up of the nodes (shared libraries, MPI, MKL): a small run, not measured.
printf "\n--- warm-up (%s^3, 1 solve, not measured) " $WARMUP_N
t0=$(date +%s)
s2l $WARMUP_N 1 1 $OUT/log.warmup "" $BASE $COARSE $STOP
printf "rc %s (%ss)" $? $(( $(date +%s) - t0 ))

for n in $SIZES; do
    d=$CASES/bub${n}_np$NP
    case_ok $d $n $NP || { printf "\n--- bub%s: no case, skipped" $n; continue; }
    STEPS=$STEPS RELTOL=$RTOL set_dicts $d
    run_s2l bub${n}_amg $n
    run_of bub${n}_$OF_SOLVER $d
done

printf "\n--- done: python3 report_scaling.py %s\n" "$(basename $OUT)"
