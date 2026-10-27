#!/bin/bash
#SBATCH --nodes=5                    	# 5 nodes = 240 processes (as the reference runs of the scaling campaign)
#SBATCH --ntasks-per-node=48         	# 48 tasks per node
#SBATCH --exclusive                  	# whole nodes: clean timings
#SBATCH --partition=g100_usr_prod    	# server type
#SBATCH --error=allrun_krylov.err    	# standard error file
#SBATCH --output=allrun_krylov.out   	# standard output file
#SBATCH --account=IscrC_EXTA-PMF       	# account name
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=francesco.buffoli@mail.polimi.it
#SBATCH --job-name=s2l_krylov
#SBATCH --time=00:35:00
# -------------------------------------------------------------------------------- #
# Krylov campaign: the two-level Schwarz preconditioner of the scaling campaign (RAS + deflated
# coarse correction, AMG local solves) inside right-preconditioned GMRES(30) and inside BiCGStab,
# on the matrices of the scaling campaign (bubbles3d: 150 bubbles, rho ratio 1000, 0.5 cells per
# step), 192^3, 256^3 and 320^3, 16 steps x 3 correctors, OpenFOAM stopping test with relTol
# RTOL = 1e-2, 240 processes.
#
# Why: in the scaling campaign orthogonalization and stopping test of GMRES were 23-28% of the time
# of the iterations (O(j) passes over the Krylov basis per iteration, plus one pass to form the
# residual for the OpenFOAM test). BiCGStab has a fixed number of vector operations per application
# of the preconditioner and explicit residuals, and accepts the nonsymmetric RAS/deflated
# preconditioner; but it does not minimise the residual, so it needs more applications. A symmetric
# preconditioner for CG is not an option: ASM needs ~1.5x the iterations of RAS on these matrices,
# more than CG saves per iteration. BiCGStab is also the solver of OpenFOAM that can host a
# nonsymmetric preconditioner (PBiCGStab + an lduMatrix::preconditioner).
#
# Variants (VARIANTS, default "amg bicgstab"):
#   amg       GMRES(30) + RAS/deflated     reference (= bub<N>_amg of the scaling campaign)
#   bicgstab  BiCGStab  + RAS/deflated     the algorithm of OpenFOAM's PBiCGStab
# Iterations of both = applications of the preconditioner (BiCGStab: two per step).
#
# GAMG: not run by default (OF_SOLVER=none): report_krylov.py takes it from the scaling campaign
# (--gamg run_scaling_np240_<jobid>, same nodes type, matrices and processes). OF_SOLVER=GAMG_DIC
# runs it here too (needs the OpenFOAM cases: Allprep_scaling.sh, or prepared here).
#
# Nothing to compile beforehand: the job builds schwarz2lvl on the compute node (cases/bin_s2l_<md5>,
# reused while the sources do not change) and checks it with test_bicgstab and test_stopping; if the
# build or a test fails it stops with "STOP: ..." in allrun_krylov.out (logs in cases/bin_s2l_<md5>/).
#   from test/gamg:      sbatch Allrun_krylov.sh            (RTOL=1e-8 sbatch --time=01:00:00 Allrun_krylov.sh: to 1e-8)
#   report:              python3 report_krylov.py run_krylov_np240_* --gamg run_scaling_np240_23040226
# Local smoke test (any MPI):  SMOKE=1 NP=4 ./Allrun_krylov.sh
# -------------------------------------------------------------------------------- #

NP=${NP:-${SLURM_NTASKS:-240}}   # processes = subdomains
SIZES=${SIZES:-"192 256 320"}    # cells per direction
STEPS=${STEPS:-16}               # time steps
CORR=${CORR:-3}                  # solves per step
RTOL=${RTOL:-1e-2}               # relTol of the normalised residual of OpenFOAM
VARIANTS=${VARIANTS:-"amg bicgstab"}
OF_SOLVER=${OF_SOLVER:-none}     # none | GAMG_DIC | ... (case/system/fvSolution)
NBUB=${NBUB:-150}; RADIUS=${RADIUS:-0.04}; RHO=${RHO:-1000}; CFL=${CFL:-0.5}; SEED=2024   # bubbles
BASE=${BASE:-"--tau 0.3 --nev 8 --eig-tol 1e-4 --local-approx amg"}   # P7b: AMG sa + Gauss-Seidel local solves
COARSE=${COARSE:-"--coarse-solver cpardiso --coarse-procs 48 --coarse-root-below 10000"}
STOP="--stop-norm openfoam --rtol $RTOL"
RUN_TIMEOUT=${RUN_TIMEOUT:-900}  # seconds: a run that does not finish is stopped, the job goes on
WARMUP_N=${WARMUP_N:-96}

variant_opts() {
    case $1 in
        amg)      echo "--krylov gmres" ;;
        bicgstab) echo "--krylov bicgstab" ;;
        *)        return 1 ;;
    esac
}

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

# Same modules as the scaling campaign also without GAMG: autoload openfoam/10 brings the toolchain
# (Intel oneAPI 2021 compilers, Intel MPI: mpicxx, mpirun, and the MPI of the MKL BLACS library), so
# the binaries and the timings are comparable with run_scaling_np240_*.
module load profile/eng
module load autoload openfoam/10
module load metis/5.1.0--gcc--10.2.0
module load intel-oneapi-mkl/2023.2.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # one sequential factorization / V-cycle per process
unset I_MPI_PMI_LIBRARY                         # silences the "will be ignored" warnings of every rank
if [ "${SMOKE:-0}" != 1 ] && [ -z "$WM_PROJECT_DIR" ]; then echo "module openfoam/10 not loaded (toolchain)"; exit 1; fi
command -v mpicxx > /dev/null && command -v mpirun > /dev/null || { echo "mpicxx / mpirun not found"; exit 1; }
for v in $VARIANTS; do variant_opts $v > /dev/null || { echo "unknown variant '$v'"; exit 1; }; done

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
REPO=$(cd ../.. && pwd)
OUT=$PWD/run_krylov_np${NP}_r${RTOL}_${SLURM_JOB_ID:-local}
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
echo "NP=$NP SIZES='$SIZES' STEPS=$STEPS CORR=$CORR RTOL=$RTOL VARIANTS='$VARIANTS' OF_SOLVER=$OF_SOLVER BASE='$BASE' COARSE='$COARSE' NBUB=$NBUB RADIUS=$RADIUS RHO=$RHO CFL=$CFL SEED=$SEED nodes=${SLURM_JOB_NUM_NODES:-1} job=${SLURM_JOB_ID:-local}" > $OUT/config.txt
# read by report_krylov.py: "<run> <kind> <options>"
{
    for v in $VARIANTS; do echo "$v schwarz2lvl $(variant_opts $v) $BASE $STOP"; done
    [ "$OF_SOLVER" != none ] && echo "$OF_SOLVER openfoam -solver $OF_SOLVER relTol $RTOL"
} > $OUT/variants.txt

# status line: NAME NP STATUS SECONDS RC
status() { printf "%s (%ss)" $3 $4; echo "$1 $2 $3 $4 $5" >> $OUT/status.txt; }

s2l() {
    local n=$1 steps=$2 corr=$3 log=$4 csv=$5; shift 5
    timeout $RUN_TIMEOUT mpirun $MPIRUN_OPTS -n $NP $BIN/schwarz2lvl --generate bubbles3d --nx $n --steps $steps \
        --correctors $corr --nbubbles $NBUB --radius $RADIUS --rho-ratio $RHO --cfl $CFL --rhs generated \
        --reuse auto "$@" ${csv:+--steps-csv $csv} > $log 2>&1
}
# run_s2l NAME N VARIANT  ->  log.NAME, csv/NAME.csv
run_s2l() {
    local name=$1 n=$2 v=$3 t0 st rc; t0=$(date +%s)
    printf "\n--- %-24s np=%-4s " "$name" "$NP"
    s2l $n $STEPS $CORR $OUT/log.$name $OUT/csv/$name.csv $(variant_opts $v) $BASE $COARSE $STOP
    rc=$?
    if   grep -q "NOT converged" $OUT/log.$name;          then st=not_converged
    elif grep -q "^Time loop summary" $OUT/log.$name;     then st=ok
    elif [ $rc = 124 ];                                   then st=timeout
    else                                                       st=error; fi
    status $name $NP $st $(( $(date +%s) - t0 )) $rc
}
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

# -------------------------------------------------------------------------------- #
# Binaries. With GAMG: those of Allprep_scaling.sh (cases/bin_<md5>) or built and checked here as in
# Allrun_scaling.sh. Without: schwarz2lvl and the tests only (cases/bin_s2l_<md5>), checked with
# test_bicgstab and test_stopping on 4 processes (~5 minutes the first time).
if [ "$OF_SOLVER" != none ]; then
    B=$CASES/bin_$SRC
    if [ ! -f $B/OK ]; then
        printf "\n--- Building and checking schwarz2lvl + bubblyPressureFoam in %s " "$B"
        build_and_check $B > $OUT/log.build_and_check 2>&1 || { printf "\nSTOP: %s\n" "$(cat $OUT/log.build_and_check)"; exit 1; }
    fi
    APP=$B/bin/bubblyPressureFoam
else
    B=$CASES/bin_s2l_$SRC
    if [ ! -f $B/OK ]; then
        printf "\n--- Building and checking schwarz2lvl in %s " "$B"
        rm -f $B/OK
        build_s2l $B || { printf "\nSTOP: build failed, see %s/log.build\n" "$B"; exit 1; }
        for t in test_bicgstab test_stopping; do
            ( cd $B && timeout 600 mpirun $MPIRUN_OPTS -n 4 $B/bin/$t > $B/log.$t 2>&1 )
            grep -q "ALL PASSED" $B/log.$t || { printf "\nSTOP: %s FAILED, see %s/log.%s\n" $t "$B" $t; exit 1; }
        done
        echo "sources=$SRC test_bicgstab=passed test_stopping=passed $(date '+%F %T')" > $B/OK
    fi
fi
printf "\n--- Binaries: %s (%s)" "$B" "$(cat $B/OK)"
cp $B/OK $OUT/binaries.txt
BIN=$B/bin

# OpenFOAM cases (only with GAMG): prepared by Allprep_scaling.sh, or here.
if [ "$OF_SOLVER" != none ]; then
    for n in $SIZES; do
        d=$CASES/bub${n}_np$NP
        if ! case_ok $d $n $NP; then
            printf "\n--- preparing %s " "$d"
            prep_case $n $NP $d ${METHOD:-scotch} > $OUT/log.prep_$n 2>&1 || printf "FAILED: %s" $OUT/log.prep_$n
        fi
    done
fi

# Warm-up of the nodes (shared libraries, MPI, MKL): a small run, not measured.
printf "\n--- warm-up (%s^3, 1 solve, not measured) " $WARMUP_N
t0=$(date +%s)
s2l $WARMUP_N 1 1 $OUT/log.warmup "" $(variant_opts amg) $BASE $COARSE $STOP
printf "rc %s (%ss)" $? $(( $(date +%s) - t0 ))

for n in $SIZES; do
    for v in $VARIANTS; do run_s2l bub${n}_$v $n $v; done
    if [ "$OF_SOLVER" != none ]; then
        d=$CASES/bub${n}_np$NP
        if case_ok $d $n $NP; then
            STEPS=$STEPS RELTOL=$RTOL set_dicts $d
            run_of bub${n}_$OF_SOLVER $d
        else
            printf "\n--- bub%s: no OpenFOAM case, GAMG skipped" $n
        fi
    fi
done

printf "\n--- done: python3 report_krylov.py %s --gamg run_scaling_np%s_<jobid>\n" "$(basename $OUT)" "$NP"
