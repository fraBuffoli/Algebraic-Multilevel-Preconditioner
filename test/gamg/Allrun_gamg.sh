#!/bin/bash
#SBATCH --nodes=5                    	# 5 nodes
#SBATCH --ntasks-per-node=48         	# 48 tasks per node -> 240 MPI processes
#SBATCH --exclusive                  	# whole nodes: clean timings
#SBATCH --partition=g100_usr_prod    	# server type
#SBATCH --error=allrun_gamg.err      	# standard error file
#SBATCH --output=allrun_gamg.out     	# standard output file
#SBATCH --account=IscrC_EXTA-PMF       	# account name
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=francesco.buffoli@mail.polimi.it
#SBATCH --job-name=s2l_gamg
#SBATCH --time=01:10:00
# -------------------------------------------------------------------------------- #
# GAMG reference: the OpenFOAM linear solvers and the two-level Schwarz preconditioner on the
# same pressure matrices (bubbles3d: 150 bubbles, rho ratio 1000, rising 0.5 cells per step).
#   schwarz2lvl (--rhs generated), run first for every size:
#     amg       P7b setting: one AMG V-cycle per local solve, --tau 0.3 --nev 8 --eig-tol 1e-4
#     exact     same with exact Cholesky local solves
#   OpenFOAM (openfoam/bubblyPressureFoam, solver controls in case/system/fvSolution):
#     GAMG_DIC  GAMG, DIC smoother (the fastest OpenFOAM variant of the first run)
#     GAMG_GS   GAMG, GaussSeidel smoother, faceAreaPair agglomeration (the usual p / p_rgh setup)
#     PCG_GAMG  CG preconditioned by 2 GAMG V-cycles (DICGaussSeidel)            (192^3 only)
#     also available (OF_A / OF_B): GAMG_ALG (algebraicPair agglomeration), PCG_DIC (CG + DIC)
# Every solve starts from 0. schwarz2lvl stops at ||r||_2 / ||b||_2 = 1e-8; OpenFOAM tests its
# normalised L1 residual, relTol 1e-9 in fvSolution so that its true 2-norm residual is also ~1e-8
# (with relTol 1e-8 it stopped at up to 1.2e-7). Times: each solve between two synchronizations
# of all processes; schwarz2lvl also its setup/update of every step.
#   bub192: 192^3, ~30k cells per process at 240 processes, 20 steps x 3 correctors
#   bub256: 256^3, ~70k cells per process at 240 processes, 16 steps x 3 correctors
# The job stops after the checks if the OpenFOAM matrix is not the generated one.
#
#   1. from test/gamg:   jid=$(sbatch --parsable Allprep_gamg.sh)   meshes (kept if already there) + checks
#   2.                   sbatch --dependency=afterok:$jid Allrun_gamg.sh    (~45 min, ~180 core-hours)
#   3. summary + plots:  python3 report_gamg.py run_gamg_<jobid>
# Local smoke test (any MPI, OpenFOAM-10 environment sourced):  SMOKE=1 NP=4 ./Allrun_gamg.sh
# -------------------------------------------------------------------------------- #

## what to run (1 = on), in this order                                    (estimates on 5 nodes)
RUN_CHECK=${RUN_CHECK:-1}        # test_bubbles + matrix check OpenFOAM vs generator (24^3, serial) ~ 1 min
RUN_A=${RUN_A:-1}                # 192^3: schwarz2lvl (~4.5 min) + OpenFOAM (~11 min)
RUN_B=${RUN_B:-1}                # 256^3: schwarz2lvl (~10 min) + OpenFOAM (~14 min)

NP=${NP:-${SLURM_NTASKS:-240}}   # processes = subdomains
RUN_TIMEOUT=${RUN_TIMEOUT:-1200} # seconds: a run that does not finish is stopped, the job goes on
N_A=${N_A:-192}; N_B=${N_B:-256}
STEPS_A=${STEPS_A:-20}; STEPS_B=${STEPS_B:-16}
CORR=${CORR:-3}                  # solves per step (PISO-like correctors)
NBUB=${NBUB:-150}; RADIUS=${RADIUS:-0.04}; RHO=${RHO:-1000}; CFL=${CFL:-0.5}; SEED=2024   # bubbles
OF_A=${OF_A:-"GAMG_DIC GAMG_GS PCG_GAMG"}
OF_B=${OF_B:-"GAMG_DIC GAMG_GS"}
S2L=${S2L:-"amg exact"}
COARSE=${COARSE:-"--coarse-solver cpardiso --coarse-procs 48 --coarse-root-below 10000"}
BASE=${BASE:-"--tau 0.3 --nev 8 --eig-tol 1e-4"}    # P5/P7b coarse space, common to the schwarz2lvl variants
OUT_PREFIX=${OUT_PREFIX:-run_gamg}

# schwarz2lvl options of a variant (added to BASE)
s2l_variant() {
    case $1 in
        amg)    echo "--local-approx amg" ;;      # P7b: AMG sa + Gauss-Seidel, coarsest <= 500
        exact)  echo "--local-approx none" ;;     # exact Cholesky factors of A_ii (P5)
        *)      echo "unknown variant $1" >&2; exit 1 ;;
    esac
}

# -------------------------------------------------------------------------------- #
S2L_CXX=${S2L_CXX:-mpicxx}
S2L_CXXFLAGS=${S2L_CXXFLAGS:-"-std=c++17 -O3 -march=skylake-avx512 -DNDEBUG"}   # schwarz2lvl and bubble field of the app
if [ "${SMOKE:-0}" = 1 ]; then        # local check of the script: tiny problems, no modules
    module() { :; }
    S2L_CXXFLAGS="-std=c++17 -O2 -DNDEBUG"
    N_A=24; N_B=32; STEPS_A=3; STEPS_B=2
    NBUB=20; RADIUS=0.1; CFL=1
    COARSE="--coarse-solver root"
    MPIRUN_OPTS=${MPIRUN_OPTS:-"--oversubscribe"}
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
OUT=$PWD/${OUT_PREFIX}_${SLURM_JOB_ID:-local}
BIN=$OUT/bin/schwarz2lvl
APP=$OUT/bin/bubblyPressureFoam
CASES=${CASES:-$PWD/cases}
[ "${SMOKE:-0}" = 1 ] && CASES=$OUT/cases
mkdir -p $OUT/csv $CASES
. ./gamg_functions.sh
module list > $OUT/log.modules 2>&1
if git -C $REPO rev-parse --git-dir > /dev/null 2>&1; then
    git -C $REPO log -1 --format='%h %s' > $OUT/log.git 2>&1
    git -C $REPO status --short -- src include test openfoam > $OUT/git_status.txt 2>&1
else
    echo "not a git checkout: see src_hash.txt" > $OUT/log.git
fi
( cd $REPO && find src include test openfoam -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.C' \) | sort | xargs md5sum ) > $OUT/src_hash.txt
echo "sources md5: $(md5sum < $OUT/src_hash.txt | cut -c1-12)" >> $OUT/log.git
echo "NP=$NP COARSE='$COARSE' BASE='$BASE' STEPS_A=$STEPS_A STEPS_B=$STEPS_B CORR=$CORR N_A=$N_A N_B=$N_B NBUB=$NBUB RADIUS=$RADIUS RHO=$RHO CFL=$CFL SEED=$SEED OpenFOAM=$WM_PROJECT_VERSION" > $OUT/config.txt
# read by report_gamg.py: "<run> <kind> <options>"
{ for v in $S2L; do echo "$v schwarz2lvl $BASE $(s2l_variant $v)"; done
  for v in $(echo $OF_A $OF_B | tr ' ' '\n' | awk '!seen[$0]++'); do echo "$v openfoam -solver $v"; done; } > $OUT/variants.txt

# status line: NAME NP STATUS SECONDS RC
status() { printf "%s (%ss)" $3 $4; echo "$1 $2 $3 $4 $5" >> $OUT/status.txt; }

# run_of NAME CASE SOLVER STEPS  ->  log.NAME, csv/NAME.csv
run_of() {
    local name=$1 dir=$2 solver=$3 steps=$4 t0 st rc; t0=$(date +%s)
    printf "\n--- %-24s np=%-4s " "$name" "$NP"
    local par=""; [ "$NP" -gt 1 ] && par="-parallel"
    timeout $RUN_TIMEOUT mpirun $MPIRUN_OPTS -n $NP $APP -case $dir $par -solver $solver -steps $steps \
        -correctors $CORR -csv $OUT/csv/$name.csv > $OUT/log.$name 2>&1
    rc=$?
    if   [ $rc = 124 ];                                                then st=timeout
    elif ! grep -q "^End" $OUT/log.$name;                              then st=error
    elif awk -F, 'NR > 1 && $11 == 0 { bad = 1 } END { exit !bad }' $OUT/csv/$name.csv; then st=not_converged
    else                                                                    st=ok; fi
    status $name $NP $st $(( $(date +%s) - t0 )) $rc
}

# run_s2l NAME N STEPS <options>  ->  log.NAME, csv/NAME.csv
run_s2l() {
    local name=$1 n=$2 steps=$3 t0 st rc; t0=$(date +%s); shift 3
    printf "\n--- %-24s np=%-4s " "$name" "$NP"
    timeout $RUN_TIMEOUT mpirun $MPIRUN_OPTS -n $NP $BIN --generate bubbles3d --nx $n --steps $steps \
        --correctors $CORR --nbubbles $NBUB --radius $RADIUS --rho-ratio $RHO --cfl $CFL --rhs generated \
        --reuse auto "$@" --steps-csv $OUT/csv/$name.csv > $OUT/log.$name 2>&1
    rc=$?
    if   grep -q "NOT converged" $OUT/log.$name;          then st=not_converged
    elif grep -q "^Time loop summary" $OUT/log.$name;     then st=ok
    elif [ $rc = 124 ];                                   then st=timeout
    else                                                       st=error; fi
    status $name $NP $st $(( $(date +%s) - t0 )) $rc
}

# -------------------------------------------------------------------------------- #
printf "\n--- Building ---"
if [ "${SMOKE:-0}" = 1 ]; then
    ( cd $REPO && make -j 4 all tests BUILD=$OUT/build BIN=$OUT/bin USE_AMGCL=1 CXX="$S2L_CXX" CXXFLAGS="$S2L_CXXFLAGS" ) > $OUT/log.build 2>&1 ||
        { printf "\nbuild FAILED: see %s/log.build\n" $OUT; exit 1; }
else
    libdir() { local d; for d in ${LD_LIBRARY_PATH//:/ }; do ls $d/$1.so* > /dev/null 2>&1 && { echo $d; return; }; done; }
    METIS_DIR=$(libdir libmetis)
    MKL_DIR=${MKLROOT:+$MKLROOT/lib/intel64}; MKL_DIR=${MKL_DIR:-$(libdir libmkl_core)}
    ls $MKL_DIR/libmkl_blacs_intelmpi_lp64.so* > /dev/null 2>&1 ||
        { printf "\nMKL BLACS for Intel MPI not found in %s\n" "$MKL_DIR"; exit 1; }
    ( cd $REPO && make -j 48 all tests BUILD=$OUT/build BIN=$OUT/bin \
        CXX="$S2L_CXX" EIGEN_INC=external/eigen3 CXXFLAGS="$S2L_CXXFLAGS" \
        METIS_INC="$(dirname "$METIS_DIR")/include" METIS_LIB="-L$METIS_DIR -Wl,-rpath,$METIS_DIR -lmetis" \
        USE_MUMPS=0 USE_UMFPACK=0 USE_PARDISO=1 USE_CPARDISO=1 USE_AMGCL=1 \
        MKL_INC="$(dirname "$(dirname "$MKL_DIR")")/include" \
        MKL_LIB="-L$MKL_DIR -Wl,-rpath,$MKL_DIR -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl" \
        MKL_BLACS_LIB="-lmkl_blacs_intelmpi_lp64" \
      ) > $OUT/log.build 2>&1 || { printf "\nbuild FAILED: see %s/log.build\n" $OUT; exit 1; }
fi
# OpenFOAM application: built in a copy (objects in $OUT/app), executable in $OUT/bin; its bubble
# field is compiled with the schwarz2lvl compiler and flags (see bubbleFieldCapi.h).
build_app $OUT > $OUT/log.build_app 2>&1 ||
    { printf "\nbuild of bubblyPressureFoam FAILED: see %s/log.build_app\n" $OUT; exit 1; }

# -------------------------------------------------------------------------------- #
if [ $RUN_CHECK = 1 ]; then
    printf "\n--- Checks ---"
    ( cd $OUT && timeout 300 mpirun $MPIRUN_OPTS -n 1 $OUT/bin/test_bubbles > $OUT/log.test_bubbles 2>&1 )
    printf "\n    %-20s %s" test_bubbles "$(tail -n1 $OUT/log.test_bubbles)"
    # Same matrix: OpenFOAM (serial 24^3 case, -writeMatrix: last step) vs the generator (matgen, t = 3).
    matrix_check $OUT $APP $OUT/bin/matgen; ok=$?
    printf "\n    %-20s %s" "matrix check" "$(tail -n1 $OUT/log.check_matrix 2>/dev/null || echo FAILED)"
    if [ $ok != 0 ]; then
        printf "\nSTOP: OpenFOAM does not solve the generated matrices (see %s/log.check_*)\n" $OUT
        exit 1
    fi
fi

prepare() {  # prepare N: mesh of size N for NP processes (normally done by Allprep_gamg.sh)
    local d=$CASES/bub$1_np$NP
    if ! case_ok $d $1 $NP; then
        printf "\n--- preparing %s here (Allprep_gamg.sh was not run: the other cores are idle) " "$d"
        prep_case $1 $NP $d ${METHOD:-scotch} > $OUT/log.prep_$1 2>&1 || { printf "FAILED: %s\n" $OUT/log.prep_$1; return 1; }
    fi
    cat $d/PREPARED >> $OUT/cases.txt
}

# A - 192^3, B - 256^3
for X in A B; do
    eval "run=\$RUN_$X n=\$N_$X steps=\$STEPS_$X ofv=\"\$OF_$X\""
    [ "$run" = 1 ] || continue
    d=$CASES/bub${n}_np$NP
    prepare $n || continue
    STEPS=$steps set_dicts $d
    for v in $S2L; do run_s2l bub${n}_$v $n $steps $BASE $(s2l_variant $v) $COARSE; done
    for v in $ofv; do run_of bub${n}_$v $d $v $steps; done
done

printf "\n--- done: python3 report_gamg.py %s\n" "$(basename $OUT)"
