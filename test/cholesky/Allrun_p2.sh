#!/bin/bash
#SBATCH --nodes=5                    	# 5 nodes
#SBATCH --ntasks-per-node=48         	# 48 tasks per node -> max 240 MPI processes
#SBATCH --exclusive                  	# whole nodes: clean timings
#SBATCH --partition=g100_usr_prod    	# server type
#SBATCH --error=allrun_p2.err         	# standard error file
#SBATCH --output=allrun_p2.out        	# standard output file
#SBATCH --account=IscrC_EXTA-PMF       	# account name
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=francesco.buffoli@mail.polimi.it
#SBATCH --job-name=s2l_p2
#SBATCH --time=01:00:00
# -------------------------------------------------------------------------------- #
# P2 - symmetric factorizations (Cholesky) and single-precision local factors.
# The same problems as the P1 campaign (Allrun_reuse.sh), with the P1 configuration as the
# reference, so that only the factorizations change:
#   lu       --local-factorization lu --coarse-factorization lu|ldlt   = P1 (LU for A_ii, A~_ii;
#            root A_00 in LU, distributed A_00 in LDL^T as cpardiso did in P1 with Intel MPI)
#   chol     defaults (auto/auto): Cholesky for A_ii, A~_ii + sigma I and A_00 (root: sequential
#            PARDISO mtype 2, cpardiso: distributed mtype 2 checked by a probe solve)
#   cholsp   chol + --local-precision single: factors of A_ii in float (one-level part only)
#   chol_cp / chol_root   chol with the other coarse solver (distributed vs rank 0)
# Time loops with --reuse auto (the production setting): the rebuild steps chosen by the
# break-even rule may differ between runs, so report_p2.py also compares the cost of each
# operation (mean refactorization, mean rebuild, solve time per iteration).
#
#   1. from test/p2:               sbatch Allrun_p2.sh
#   2. summary:                    python3 report_p2.py run_p2_<jobid>
# The binary is built inside the run directory (does not touch ../../bin).
# Local smoke test of this script (small sizes, any MPI):
#   SMOKE=1 NP=4 ./Allrun_p2.sh
# -------------------------------------------------------------------------------- #

## what to run (1 = on), in this order                                     (estimates on 5 nodes)
RUN_CHECK=${RUN_CHECK:-1}        # build + unit tests on 16 processes (incl. test_factorization)   ~ 7 min
RUN_SEQ=${RUN_SEQ:-1}            # bubbles3d 128^3, rho 1000, root A_00: lu/chol/cholsp/chol_cp    ~ 7 min
RUN_LARGE=${RUN_LARGE:-1}        # bubbles3d 192^3, cpardiso A_00: lu/chol/cholsp/chol_root        ~ 16 min
RUN_PAPER=${RUN_PAPER:-1}        # single solves of SPD SuiteSparse matrices (if in ../../matrices) ~ 5 min

NP=${NP:-240}                    # processes = subdomains
RUN_TIMEOUT=${RUN_TIMEOUT:-900}  # seconds: a run that does not finish is stopped, the job goes on
COARSE=${COARSE:-"--coarse-solver cpardiso --coarse-procs 48 --coarse-root-below 10000"}
STEPS=${STEPS:-30}               # time steps of the 128^3 runs (P1: 60)
STEPS_LARGE=${STEPS_LARGE:-30}   # 192^3 runs (as P1)
CORR=${CORR:-3}                  # correctors per step (PISO-like)
N_SEQ=${N_SEQ:-128}; N_LARGE=${N_LARGE:-192}
# Bubbles: radius 0.04 = 5 cells at 128^3, ~4% gas; subdomains of ~20 cells: bubbles cross the interfaces.
BUB="--generate bubbles3d --nbubbles 150 --radius 0.04 --cfl 0.5 --tau 0.3 --nev 60 --rho-ratio 1000 --reuse auto"
# SPD matrices of Table 1 where the cpardiso factorization of A_00 is a large part of the setup.
PAPER_MATRICES=${PAPER_MATRICES:-"G3_circuit thermal2 ecology2"}
PAPER="--tau 0.6 --nev 300 --maxit 200 --coarse-solver cpardiso --coarse-procs 48"

LU_ROOT="--local-factorization lu --coarse-factorization lu"     # P1 with the root coarse solver
LU_CP="--local-factorization lu --coarse-factorization ldlt"     # P1 with cpardiso (Intel MPI: mtype -2)
CHOL=""                                                          # P2 defaults: auto / auto
SP="--local-precision single"

# -------------------------------------------------------------------------------- #
if [ "${SMOKE:-0}" = 1 ]; then        # local check of the script: tiny problems, no modules
    module() { :; }
    N_SEQ=24; N_LARGE=28; STEPS=4; STEPS_LARGE=3
    BUB="--generate bubbles3d --nbubbles 20 --radius 0.1 --cfl 1 --tau 0.3 --nev 20 --rho-ratio 1000 --reuse auto"
    COARSE="--coarse-solver cpardiso --coarse-procs 2 --coarse-root-below 40"
    PAPER="--tau 0.6 --nev 30 --maxit 200 --coarse-solver cpardiso --coarse-procs 2"
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
OUT=$PWD/run_p2_${SLURM_JOB_ID:-local}
BIN=$OUT/bin/schwarz2lvl
mkdir -p $OUT/csv $OUT/single
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
echo "NP=$NP COARSE='$COARSE' STEPS=$STEPS STEPS_LARGE=$STEPS_LARGE CORR=$CORR N_SEQ=$N_SEQ N_LARGE=$N_LARGE" > $OUT/config.txt

# status NAME NP T0 RC: classifies a finished run from its log
status() {
    local name=$1 np=$2 t0=$3 rc=$4 st
    if   grep -q "NOT converged" $OUT/log.$name;                        then st=not_converged
    elif grep -qE "^Time loop summary|^  converged in" $OUT/log.$name;  then st=ok
    elif [ $rc = 124 ];                                                 then st=timeout
    else                                                                     st=error; fi
    printf "%s (%ss)" $st $(( $(date +%s) - t0 ))
    echo "$name $np $st $(( $(date +%s) - t0 )) $rc" >> $OUT/status.txt
}

# run NAME NP <schwarz2lvl options>  ->  time loop: log.NAME, csv/NAME.csv (one line per solve)
run() {
    local name=$1 np=$2 t0 rc; t0=$(date +%s); shift 2
    printf "\n--- %-28s np=%-4s " "$name" "$np"
    timeout $RUN_TIMEOUT mpirun $MPIRUN_OPTS -n $np $BIN "$@" --correctors $CORR \
        --steps-csv $OUT/csv/$name.csv > $OUT/log.$name 2>&1
    rc=$?; status $name $np $t0 $rc
}

# run_single NAME NP <schwarz2lvl options>  ->  one solve: log.NAME, single/NAME.csv
run_single() {
    local name=$1 np=$2 t0 rc; t0=$(date +%s); shift 2
    printf "\n--- %-28s np=%-4s " "$name" "$np"
    timeout $RUN_TIMEOUT mpirun $MPIRUN_OPTS -n $np $BIN "$@" --csv $OUT/single/$name.csv > $OUT/log.$name 2>&1
    rc=$?; status $name $np $t0 $rc
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
    for t in test_subdomain test_preconditioner test_reuse test_factorization; do
        ( cd $OUT && timeout $RUN_TIMEOUT mpirun $MPIRUN_OPTS -n $NCHECK $OUT/bin/$t > $OUT/log.$t 2>&1 )
        printf "\n    %-20s %s" $t "$(tail -n1 $OUT/log.$t)"
    done
    # test_factorization also checks the distributed Cholesky of cpardiso (mtype 2 + probe solve),
    # which cannot be tested with Open MPI. If it fails, the runs go on with the P1 coarse
    # factorization (LDL^T), so that the local Cholesky is still measured.
    if ! grep -q "ALL PASSED" $OUT/log.test_factorization; then
        printf "\ntest_factorization FAILED: the runs use --coarse-factorization ldlt (see log.test_factorization)"
        CHOL="--coarse-factorization ldlt"
        echo "test_factorization failed: CHOL='$CHOL'" >> $OUT/config.txt
    fi
fi

# SEQ - n0 ~ 6000 < 10000: A_00 on rank 0 (P1: half of the time in the coarse solves and
# refactorizations). chol_cp forces cpardiso: which coarse solver is better with Cholesky?
if [ $RUN_SEQ = 1 ]; then
    run bub${N_SEQ}_lu       $NP $BUB --nx $N_SEQ --steps $STEPS $COARSE $LU_ROOT
    run bub${N_SEQ}_chol     $NP $BUB --nx $N_SEQ --steps $STEPS $COARSE $CHOL
    run bub${N_SEQ}_cholsp   $NP $BUB --nx $N_SEQ --steps $STEPS $COARSE $CHOL $SP
    run bub${N_SEQ}_chol_cp  $NP $BUB --nx $N_SEQ --steps $STEPS $COARSE $CHOL --coarse-root-below 0
fi

# LARGE - subdomains of ~30k cells, n0 ~ 12600: cpardiso (P1: local solves 25%, coarse 26%).
# chol_root moves A_00 to rank 0.
if [ $RUN_LARGE = 1 ]; then
    run bub${N_LARGE}_lu        $NP $BUB --nx $N_LARGE --steps $STEPS_LARGE $COARSE $LU_CP
    run bub${N_LARGE}_chol      $NP $BUB --nx $N_LARGE --steps $STEPS_LARGE $COARSE $CHOL
    run bub${N_LARGE}_cholsp    $NP $BUB --nx $N_LARGE --steps $STEPS_LARGE $COARSE $CHOL $SP
    run bub${N_LARGE}_chol_root $NP $BUB --nx $N_LARGE --steps $STEPS_LARGE $COARSE $CHOL --coarse-root-below 1000000
fi

# PAPER - one solve (setup + GMRES) of SPD matrices: setup dominated by the eigensolver and by
# the distributed factorization of A_00 (n0 = 13k-25k).
if [ $RUN_PAPER = 1 ]; then
    MAT=$REPO/matrices
    if [ "${SMOKE:-0}" = 1 ]; then
        run_single lap40_lu   $NP --generate laplace3d --nx 40 $PAPER $LU_CP
        run_single lap40_chol $NP --generate laplace3d --nx 40 $PAPER $CHOL
    fi
    for m in $PAPER_MATRICES; do
        [ -f $MAT/$m.mtx ] || { printf "\n--- %s: missing (test/paper/get_matrices.sh)" $m; continue; }
        run_single ${m}_lu   $NP --matrix $MAT/$m.mtx $PAPER $LU_CP
        run_single ${m}_chol $NP --matrix $MAT/$m.mtx $PAPER $CHOL
    done
fi

printf "\n--- done: python3 report_p2.py %s\n" "$(basename $OUT)"
