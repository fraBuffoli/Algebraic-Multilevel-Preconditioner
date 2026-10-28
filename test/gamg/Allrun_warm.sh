#!/bin/bash
#SBATCH --nodes=5                    	# 5 nodes = 240 processes (submit_warm.sh: --nodes 5 and 9)
#SBATCH --ntasks-per-node=48         	# 48 tasks per node
#SBATCH --exclusive                  	# whole nodes: clean timings
#SBATCH --partition=g100_usr_prod    	# server type
#SBATCH --error=allrun_warm.err      	# standard error file
#SBATCH --output=allrun_warm.out     	# standard output file
#SBATCH --account=IscrC_EXTA-PMF       	# account name
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=francesco.buffoli@mail.polimi.it
#SBATCH --job-name=s2l_warm
#SBATCH --time=01:30:00
# -------------------------------------------------------------------------------- #
# Warm-start campaign (phase A before a port): GAMG of OpenFOAM against the two-level Schwarz
# preconditioner (setting to port) and the one-level RAS, in the conditions of a PIMPLE time loop
# of a VoF solver instead of those of the earlier campaigns:
#
#   earlier campaigns                          this campaign
#   b = vol x alpha (constant on the bubbles)  b = -div(phig): p_rgh source of interFoam for the fluid
#                                              at rest, interface fluxes only (SOURCE=buoyancy)
#   every solve from 0                         every solve from the previous solution (WARM=1)
#   3 identical solves per step, relTol 1e-2   correctors: relTol RTOL = 0.05 or tolerance ATOL = 1e-7;
#                                              last corrector of the step: tolerance 1e-7 only
#                                              (relTol 0, as p_rghFinal)
#
# Same matrices as before (bubbles3d: 150 bubbles, rho ratio 1000, rising 0.5 cells per step), the
# right-hand side is the only change in the systems. Both codes stop on the normalised residual of
# OpenFOAM, sum|b - A x| / F (F from the initial guess), at max(relTol x initial, tolerance):
#   OpenFOAM     solvers W_<variant> (correctors) and W_<variant>Final (last corrector) of
#                case/system/fvSolution, bubblyPressureFoam -warmStart
#   schwarz2lvl  --stop-norm openfoam --rtol RTOL --atol ATOL --final-rtol 0 --final-atol ATOL
#                --warm-start 1
# Variants:
#   s2l_2L   BiCGStab + RAS/deflated two-level, AMG local solves, --tau 0.3 --nev 8 --eig-tol 1e-2
#            --eig-ncv 12, reuse auto (the setting to port, RESULTS.md section 7)
#   s2l_1L   BiCGStab + one-level RAS, AMG local solves (no coarse space: no eigensolver, no rebuilds)
#   W_GAMG   GAMG, DIC smoother (correctors and last corrector)
#   W_GAMG_PCG  GAMG for the correctors, PCG + GAMG for the last corrector: only at the sizes
#            OF_EXTRA_SIZES (default: the smallest; PCG + GAMG was 2.6x slower than GAMG to 1e-8)
# Times of every solve between two synchronizations of all processes (csv/<run>.csv), as before.
#
# Nothing to compile beforehand: the job builds schwarz2lvl, the tests, matgen and bubblyPressureFoam
# (cases/bin_<sources md5>, shared with Allprep_scaling.sh; a lock keeps two jobs from building at
# the same time), checks the matrices of both right-hand sides against OpenFOAM (24^3) and runs
# test_warmstart, test_stopping and test_bicgstab on 4 processes. Missing OpenFOAM cases are prepared
# here (Allprep_scaling.sh prepares them on one node beforehand). Any failure: "STOP: ..." in
# allrun_warm*.out.
#
#   1. from test/gamg:   ./submit_warm.sh       (Allprep_scaling.sh + this script on 5 and 9 nodes)
#                        or: sbatch Allrun_warm.sh   (240 processes, everything prepared here)
#   2. report:           python3 report_warm.py run_warm_np*_<jobid>
# Separating the two effects (optional, one more job each):
#   SOURCE=alpha sbatch Allrun_warm.sh       old right-hand side, warm start + PIMPLE tolerances
#   WARM=0 sbatch Allrun_warm.sh             physical right-hand side, every solve from 0
# Local smoke test (any MPI, OpenFOAM-10 environment sourced):  SMOKE=1 NP=1 ./Allrun_warm.sh
# -------------------------------------------------------------------------------- #

NP=${NP:-${SLURM_NTASKS:-240}}   # processes = subdomains
SIZES=${SIZES:-"192 256 320"}    # cells per direction
STEPS=${STEPS:-24}               # time steps (3 rebuild cycles of the coarse space at 256^3 / 320^3)
CORR=${CORR:-3}                  # PISO correctors per step (interFoam tutorials: nCorrectors 3)
SOURCE=${SOURCE:-buoyancy}       # right-hand side: buoyancy (p_rgh source) | alpha (earlier campaigns)
WARM=${WARM:-1}                  # 1: every solve from the previous solution (OpenFOAM); 0: from 0
RTOL=${RTOL:-0.05}               # relTol of the correctors (p_rgh)
ATOL=${ATOL:-1e-7}               # tolerance of every corrector; the only test of the last one (p_rghFinal)
S2L_VARIANTS=${S2L_VARIANTS:-"s2l_2L s2l_1L"}
OF_VARIANTS=${OF_VARIANTS:-"W_GAMG"}
OF_EXTRA=${OF_EXTRA:-"W_GAMG_PCG"}       # OpenFOAM variants run only at OF_EXTRA_SIZES
OF_EXTRA_SIZES=${OF_EXTRA_SIZES:-"${SIZES%% *}"}
NBUB=${NBUB:-150}; RADIUS=${RADIUS:-0.04}; RHO=${RHO:-1000}; CFL=${CFL:-0.5}; SEED=2024   # bubbles
COARSE=${COARSE:-"--coarse-solver cpardiso --coarse-procs 48 --coarse-root-below 10000"}
RUN_TIMEOUT=${RUN_TIMEOUT:-1500} # seconds: a run that does not finish is stopped, the job goes on
WARMUP_N=${WARMUP_N:-96}
OUT_PREFIX=${OUT_PREFIX:-run_warm}

variant_opts() {
    case $1 in
        s2l_2L) echo "--krylov bicgstab --tau 0.3 --nev 8 --eig-tol 1e-2 --eig-ncv 12 --local-approx amg $COARSE" ;;
        s2l_1L) echo "--krylov bicgstab --coarse none --local-approx amg" ;;
        *)      return 1 ;;
    esac
}
STOP="--stop-norm openfoam --rtol $RTOL --atol $ATOL --final-rtol 0 --final-atol $ATOL --warm-start $WARM"
OF_WARM=""; [ "$WARM" = 1 ] && OF_WARM="-warmStart"

# -------------------------------------------------------------------------------- #
S2L_CXX=${S2L_CXX:-mpicxx}
S2L_CXXFLAGS=${S2L_CXXFLAGS:-"-std=c++17 -O3 -march=skylake-avx512 -DNDEBUG"}   # as in Allprep_scaling.sh
if [ "${SMOKE:-0}" = 1 ]; then        # local check of the script: tiny problems, no modules
    module() { :; }
    S2L_CXXFLAGS="-std=c++17 -O2 -DNDEBUG"
    SIZES=${SMOKE_SIZES:-"16 20"}; STEPS=4; WARMUP_N=12
    OF_EXTRA_SIZES="${SIZES%% *}"
    NBUB=20; RADIUS=0.1; CFL=1
    COARSE="--coarse-solver root"
    MPIRUN_OPTS=${MPIRUN_OPTS:-"--oversubscribe"}
    CASES=${CASES:-$PWD/cases_smoke}
    METHOD=${METHOD:-hierarchical}
fi

module load profile/eng
module load autoload openfoam/10
module load metis/5.1.0--gcc--10.2.0
module load intel-oneapi-mkl/2023.2.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # one sequential factorization / V-cycle per process
unset I_MPI_PMI_LIBRARY                         # silences the "will be ignored" warnings of every rank
[ -n "$WM_PROJECT_DIR" ] || { echo "OpenFOAM environment not loaded"; exit 1; }
for v in $S2L_VARIANTS; do variant_opts $v > /dev/null || { echo "unknown variant '$v'"; exit 1; }; done
case $SOURCE in alpha|buoyancy) ;; *) echo "SOURCE must be alpha or buoyancy"; exit 1 ;; esac

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
REPO=$(cd ../.. && pwd)
TAG=""; [ "$SOURCE" != buoyancy ] && TAG="${TAG}_$SOURCE"; [ "$WARM" != 1 ] && TAG="${TAG}_cold"
OUT=$PWD/${OUT_PREFIX}${TAG}_np${NP}_${SLURM_JOB_ID:-local}
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
echo "NP=$NP SIZES='$SIZES' STEPS=$STEPS CORR=$CORR SOURCE=$SOURCE WARM=$WARM RTOL=$RTOL ATOL=$ATOL S2L_VARIANTS='$S2L_VARIANTS' OF_VARIANTS='$OF_VARIANTS' OF_EXTRA='$OF_EXTRA' OF_EXTRA_SIZES='$OF_EXTRA_SIZES' COARSE='$COARSE' NBUB=$NBUB RADIUS=$RADIUS RHO=$RHO CFL=$CFL SEED=$SEED OpenFOAM=$WM_PROJECT_VERSION nodes=${SLURM_JOB_NUM_NODES:-1} job=${SLURM_JOB_ID:-local}" > $OUT/config.txt
# read by report_warm.py: "<variant> <kind> <options>"
{
    for v in $S2L_VARIANTS; do echo "$v schwarz2lvl $(variant_opts $v) $STOP"; done
    for o in $OF_VARIANTS $OF_EXTRA; do echo "$o openfoam -solver $o $OF_WARM relTol $RTOL tolerance $ATOL, ${o}Final tolerance $ATOL"; done
} > $OUT/variants.txt

# status line: NAME NP STATUS SECONDS RC
status() { printf "%s (%ss)" $3 $4; echo "$1 $2 $3 $4 $5" >> $OUT/status.txt; }

s2l() {
    local n=$1 steps=$2 corr=$3 log=$4 csv=$5; shift 5
    timeout $RUN_TIMEOUT mpirun $MPIRUN_OPTS -n $NP $BIN/schwarz2lvl --generate bubbles3d --nx $n --steps $steps \
        --correctors $corr --nbubbles $NBUB --radius $RADIUS --rho-ratio $RHO --cfl $CFL --rhs generated \
        --bubble-source $SOURCE --reuse auto "$@" ${csv:+--steps-csv $csv} > $log 2>&1
}
# run_s2l NAME N VARIANT  ->  log.NAME, csv/NAME.csv
run_s2l() {
    local name=$1 n=$2 v=$3 t0 st rc; t0=$(date +%s)
    printf "\n--- %-24s np=%-4s " "$name" "$NP"
    s2l $n $STEPS $CORR $OUT/log.$name $OUT/csv/$name.csv $(variant_opts $v) $STOP
    rc=$?
    if   grep -q "NOT converged" $OUT/log.$name;          then st=not_converged
    elif grep -q "^Time loop summary" $OUT/log.$name;     then st=ok
    elif [ $rc = 124 ];                                   then st=timeout
    else                                                       st=error; fi
    status $name $NP $st $(( $(date +%s) - t0 )) $rc
}
# run_of NAME CASE SOLVER  ->  log.NAME, csv/NAME.csv
run_of() {
    local name=$1 dir=$2 solver=$3 t0 st rc; t0=$(date +%s)
    printf "\n--- %-24s np=%-4s " "$name" "$NP"
    local par=""; [ "$NP" -gt 1 ] && par="-parallel"
    timeout $RUN_TIMEOUT mpirun $MPIRUN_OPTS -n $NP $APP -case $dir $par -solver $solver -steps $STEPS \
        -correctors $CORR $OF_WARM -csv $OUT/csv/$name.csv > $OUT/log.$name 2>&1
    rc=$?
    if   [ $rc = 124 ];                                                then st=timeout
    elif ! grep -q "^End" $OUT/log.$name;                              then st=error
    elif awk -F, 'NR > 1 && $11 == 0 { bad = 1 } END { exit !bad }' $OUT/csv/$name.csv; then st=not_converged
    else                                                                    st=ok; fi
    status $name $NP $st $(( $(date +%s) - t0 )) $rc
}

# -------------------------------------------------------------------------------- #
# Binaries (cases/bin_<md5>): built and checked once per version of the sources, under a lock (the
# jobs of submit_warm.sh may start together). A directory built by Allprep_scaling.sh with these
# sources (both right-hand sides checked) only needs the unit tests of this campaign.
B=$CASES/bin_$SRC
printf "\n--- Binaries %s " "$B"
(
    flock -w 5400 9 || { echo "lock $CASES/bin_$SRC.lock not obtained"; exit 1; }
    if [ ! -f $B/OK ] || ! grep -q "both sources" $B/OK; then
        rm -rf $B && mkdir -p $B
        build_and_check $B > $B/log.build_and_check 2>&1 || { cat $B/log.build_and_check; exit 1; }
    fi
    if ! grep -q "test_warmstart=passed" $B/OK; then
        for t in test_warmstart test_stopping test_bicgstab; do
            ( cd $B && timeout 900 mpirun $MPIRUN_OPTS -n 4 $B/bin/$t > $B/log.$t 2>&1 )
            grep -q "ALL PASSED" $B/log.$t || { echo "$t FAILED, see $B/log.$t"; exit 1; }
        done
        echo "$(cat $B/OK) test_warmstart=passed test_stopping=passed test_bicgstab=passed" > $B/OK.new && mv $B/OK.new $B/OK
    fi
) 9> $CASES/bin_$SRC.lock > $OUT/log.binaries 2>&1 || { printf "\nSTOP: %s\n" "$(tail -n 3 $OUT/log.binaries)"; exit 1; }
printf "%s" "$(cat $B/OK)"
cp $B/OK $OUT/binaries.txt
cp $B/log.check_matrix $OUT/log.check_matrix 2>/dev/null
BIN=$B/bin
APP=$B/bin/bubblyPressureFoam

# Cases: prepared by Allprep_scaling.sh (same names); a missing one is prepared here (~5-15 minutes).
for n in $SIZES; do
    d=$CASES/bub${n}_np$NP
    if ! case_ok $d $n $NP; then
        printf "\n--- preparing %s here " "$d"
        prep_case $n $NP $d ${METHOD:-scotch} > $OUT/log.prep_$n 2>&1 || printf "FAILED: %s" $OUT/log.prep_$n
    fi
    case_ok $d $n $NP && cat $d/PREPARED >> $OUT/cases.txt
done

# Warm-up of the nodes (shared libraries, MPI, MKL): a small run, not measured.
printf "\n--- warm-up (%s^3, 1 solve, not measured) " $WARMUP_N
t0=$(date +%s)
s2l $WARMUP_N 1 1 $OUT/log.warmup "" $(variant_opts s2l_2L) $STOP
printf "rc %s (%ss)" $? $(( $(date +%s) - t0 ))

for n in $SIZES; do
    for v in $S2L_VARIANTS; do run_s2l bub${n}_$v $n $v; done
    d=$CASES/bub${n}_np$NP
    if ! case_ok $d $n $NP; then
        printf "\n--- bub%s: no OpenFOAM case, OpenFOAM skipped" $n
        continue
    fi
    STEPS=$STEPS SOURCE=$SOURCE W_RTOL=$RTOL W_ATOL=$ATOL set_dicts $d ||
        { printf "\n--- bub%s: set_dicts failed, OpenFOAM skipped" $n; continue; }
    extra=""; case " $OF_EXTRA_SIZES " in *" $n "*) extra=$OF_EXTRA ;; esac
    for o in $OF_VARIANTS $extra; do run_of bub${n}_$o $d $o; done
done

printf "\n--- done: python3 report_warm.py %s\n" "$(basename $OUT)"
