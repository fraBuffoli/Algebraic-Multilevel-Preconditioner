#!/bin/bash
#SBATCH --nodes=20                   	# 20 nodes
#SBATCH --ntasks-per-node=48         	# 48 tasks per node -> 960 MPI processes
#SBATCH --exclusive                  	# whole nodes: clean timings
#SBATCH --partition=g100_usr_prod    	# server type
#SBATCH --error=allrun_p5b.err        	# standard error file
#SBATCH --output=allrun_p5b.out       	# standard output file
#SBATCH --account=IscrC_EXTA-PMF       	# account name
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=francesco.buffoli@mail.polimi.it
#SBATCH --job-name=s2l_p5b
#SBATCH --time=01:30:00
# -------------------------------------------------------------------------------- #
# P5b - second campaign after P5: with few coarse vectors per subdomain (--nev 16 / 8) the
# solves with the factors of A_ii became the largest cost and the eigensolver is ~2/3 of a
# rebuild. Two existing options attack them, alone and together:
#   single precision factors of A_ii (--local-precision single, P2)
#   looser eigensolver tolerance     (--eig-tol 1e-4 instead of 1e-8)
# at 960 processes (n0 = sum m_i grows 4x with respect to P5), on two problems:
#   bub192: the 192^3 problem of P5 (~7k cells per process: strong scaling from 240 processes)
#   bub256: 256^3 (16.8M cells, ~17k per process), same bubbles
# No code change; requires P2 (single precision) and P5 (report_p5.py reads variants.txt).
#
#   1. from test/p5:               sbatch Allrun_p5b.sh
#      other sizes:                NP=1200 sbatch --nodes=25 Allrun_p5b.sh
#   2. summary + plot:             python3 report_p5.py run_p5b_<jobid>
# Local smoke test of this script (small sizes, any MPI):
#   SMOKE=1 NP=4 ./Allrun_p5b.sh
# -------------------------------------------------------------------------------- #

## what to run (1 = on), in this order                                   (estimates on 20 nodes)
RUN_CHECK=${RUN_CHECK:-0}        # unit tests (same code as P2: off by default)
RUN_A=${RUN_A:-1}                # bubbles3d 192^3, 8 variants, 30 steps                       ~ 15 min
RUN_B=${RUN_B:-1}                # bubbles3d 256^3, 8 variants, 24 steps                       ~ 30 min

NP=${NP:-${SLURM_NTASKS:-960}}   # processes = subdomains
RUN_TIMEOUT=${RUN_TIMEOUT:-900}  # seconds: a run that does not finish is stopped, the job goes on
COARSE=${COARSE:-"--coarse-solver cpardiso --coarse-procs 48 --coarse-root-below 10000"}
STEPS_A=${STEPS_A:-30}           # time steps, 192^3
STEPS_B=${STEPS_B:-24}           # time steps, 256^3 (rank 0 generates and sends 16.8M values per step)
CORR=${CORR:-3}                  # correctors per step (PISO-like)
N_A=${N_A:-192}; N_B=${N_B:-256}
BUB="--generate bubbles3d --nbubbles 150 --radius 0.04 --cfl 0.5 --rho-ratio 1000 --reuse auto"
# ref = P1/P2 setting; n16/n8 = best of P5; sp = single-precision A_ii; e4 = --eig-tol 1e-4.
VARIANTS=${VARIANTS:-"ref n16 n16sp n16e4 n16spe4 n8 n8sp n8spe4"}

# coarse-space / local-solver options of a variant
variant() {
    case $1 in
        ref)     echo "--tau 0.3 --nev 60" ;;
        n16)     echo "--tau 0.3 --nev 16" ;;
        n16sp)   echo "--tau 0.3 --nev 16 --local-precision single" ;;
        n16e4)   echo "--tau 0.3 --nev 16 --eig-tol 1e-4" ;;
        n16spe4) echo "--tau 0.3 --nev 16 --local-precision single --eig-tol 1e-4" ;;
        n8)      echo "--tau 0.3 --nev 8" ;;
        n8sp)    echo "--tau 0.3 --nev 8 --local-precision single" ;;
        n8e4)    echo "--tau 0.3 --nev 8 --eig-tol 1e-4" ;;
        n8spe4)  echo "--tau 0.3 --nev 8 --local-precision single --eig-tol 1e-4" ;;
        *)       echo "unknown variant $1" >&2; exit 1 ;;
    esac
}

# -------------------------------------------------------------------------------- #
if [ "${SMOKE:-0}" = 1 ]; then        # local check of the script: tiny problems, no modules
    module() { :; }
    N_A=24; N_B=28; STEPS_A=4; STEPS_B=3
    BUB="--generate bubbles3d --nbubbles 20 --radius 0.1 --cfl 1 --rho-ratio 1000 --reuse auto"
    COARSE="--coarse-solver cpardiso --coarse-procs 2 --coarse-root-below 40"
    MPIRUN_OPTS=${MPIRUN_OPTS:-"--oversubscribe"}; NCHECK=${NCHECK:-4}
fi
NCHECK=${NCHECK:-16}

module load profile/eng
module load autoload openfoam/10
module load metis/5.1.0--gcc--10.2.0
module load intel-oneapi-mkl/2023.2.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # one sequential factorization per process
unset I_MPI_PMI_LIBRARY                         # silences the "will be ignored" warnings of every rank

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
REPO=$(cd ../.. && pwd)
OUT=$PWD/run_p5b_${SLURM_JOB_ID:-local}
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
echo "NP=$NP COARSE='$COARSE' STEPS_A=$STEPS_A STEPS_B=$STEPS_B CORR=$CORR N_A=$N_A N_B=$N_B" > $OUT/config.txt
for v in $VARIANTS; do echo "$v $(variant $v)"; done > $OUT/variants.txt   # read by report_p5.py

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
timeout 120 mpirun $MPIRUN_OPTS -n 1 $BIN --help 2>/dev/null | grep -q -- "--local-precision" ||
    { printf "\nthis build has no --local-precision: P2 is required\n"; exit 1; }

if [ $RUN_CHECK = 1 ]; then
    printf "\n--- Unit tests (%s processes) ---" $NCHECK
    for t in test_subdomain test_preconditioner test_reuse test_factorization; do
        ( cd $OUT && timeout $RUN_TIMEOUT mpirun $MPIRUN_OPTS -n $NCHECK $OUT/bin/$t > $OUT/log.$t 2>&1 )
        printf "\n    %-20s %s" $t "$(tail -n1 $OUT/log.$t)"
    done
fi

# A - the 192^3 problem of P5 on 4x the processes (strong scaling): smaller subdomains, n0 4x.
if [ $RUN_A = 1 ]; then
    for v in $VARIANTS; do
        run bub${N_A}_$v $NP $BUB --nx $N_A --steps $STEPS_A $(variant $v) $COARSE
    done
fi

# B - finer mesh of the same flow (16.8M cells): subdomains of ~17k cells at 960 processes.
if [ $RUN_B = 1 ]; then
    for v in $VARIANTS; do
        run bub${N_B}_$v $NP $BUB --nx $N_B --steps $STEPS_B $(variant $v) $COARSE
    done
fi

printf "\n--- done: python3 report_p5.py %s\n" "$(basename $OUT)"
