#!/bin/bash
#SBATCH --nodes=5                    	# 5 nodes
#SBATCH --ntasks-per-node=48         	# 48 tasks per node -> max 240 MPI processes
#SBATCH --exclusive                  	# whole nodes: clean timings
#SBATCH --partition=g100_usr_prod    	# server type
#SBATCH --error=allrun_perf.err       	# standard error file
#SBATCH --output=allrun_perf.out      	# standard output file
#SBATCH --account=IscrC_EXTA-PMF       	# account name
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=francesco.buffoli@mail.polimi.it
#SBATCH --job-name=s2l_perf
#SBATCH --time=01:30:00
# -------------------------------------------------------------------------------- #
# Performance campaign after the first optimizations of the setup and of the solve:
#   - PARDISO without iterative refinement (local solver and cpardiso, iparm[7] = 0);
#   - kernel detection of A~_ii with multi-RHS solves (one per inverse iteration step);
#   - timers of the preconditioner application (coarse restrict/solve/prolong, spmv, local halo/lu);
#   - --coarse-root-below N: root coarse solver when n0 < N.
#
#   1. from test/paper:            sbatch Allrun_perf.sh
#   2. before -> after:            python3 plot_paper.py run_perf_<jobid> --ref run_22843541
#      (REF runs have the same names as in run_22843541: n0 and iterations must not change)
#   3. root vs cpardiso:           run_perf_<jobid>/cal_summary.txt  -> value of --coarse-root-below
# The binary is built inside the run directory (does not touch ../../bin).
# -------------------------------------------------------------------------------- #

## what to run (1 = on), in this order
RUN_CHECK=1          # build + MPI unit tests on 16 processes                                 ~ 5 min
RUN_REF=1            # subset of run_22843541 with the same names and options               ~ 15 min
RUN_CAL=1            # root vs cpardiso on the same problems, n0 = 0.7k ... 25k              ~ 10 min
RUN_THR=1            # finan512, tau = 0.6 (n0 = 109, crashed with cpardiso) with the switch  < 1 min

NP=240               # processes = subdomains
RUN_TIMEOUT=1200     # seconds: a run that does not finish is stopped, the job goes on
COARSE="--coarse-solver cpardiso --coarse-procs 48"   # as in run_22843541
THR=5000             # --coarse-root-below used by RUN_THR (to be replaced by the value from cal_summary.txt)

# -------------------------------------------------------------------------------- #
module load profile/eng
module load autoload openfoam/10
module load metis/5.1.0--gcc--10.2.0
module load intel-oneapi-mkl/2023.2.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # one sequential LU per process

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
REPO=$(cd ../.. && pwd)
OUT=$PWD/run_perf_${SLURM_JOB_ID:-local}
BIN=$OUT/bin/schwarz2lvl
MAT=$REPO/matrices
mkdir -p $OUT/csv $OUT/hist
module list > $OUT/log.modules 2>&1
git -C $REPO log -1 --format='%h %s' > $OUT/log.git 2>&1
git -C $REPO status --short -- src include > $OUT/git_status.txt 2>&1   # uncommitted changes (not log.*: plot_paper.py reads log.* as runs)
echo "NP=$NP COARSE='$COARSE' THR=$THR" > $OUT/config.txt

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

# -------------------------------------------------------------------------------- #
printf "\n--- Building (USE_CPARDISO=1) ---"
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

if [ $RUN_CHECK = 1 ]; then
    printf "\n--- Unit tests ---"
    mpirun -n 16 $OUT/bin/test_subdomain      > $OUT/log.test_subdomain 2>&1
    mpirun -n 16 $OUT/bin/test_preconditioner > $OUT/log.test_preconditioner 2>&1
    printf "\n    test_subdomain: %s   test_preconditioner: %s" \
        "$(tail -n1 $OUT/log.test_subdomain)" "$(tail -n1 $OUT/log.test_preconditioner)"
fi

# REF - same names and options as run_22843541 (Allrun.sh), so that plot_paper.py --ref pairs them.
#   fig4: large subdomains, eigensolver dominated by the sparse solves (target of iparm[7] = 0);
#   tab2: small subdomains, n0 = 2.3k / 7.4k;
#   tab1: G3_circuit (tau 0.6 and 0.9: A_00 factorization dominant at 0.9), memchip (dense Krylov
#         work dominant), CoupCons3D and nxp1 (one slow subdomain: load imbalance).
if [ $RUN_REF = 1 ]; then
    for N in 30 60 120 240; do
        run fig4_N$N $N --generate convdiff3d --nx 200 --nu 1e-2 --rhs generated --tau 0.3 --nev 60 $COARSE
    done
    run tab2_3d_nu1e-2_2L $NP --generate convdiff3d --nx 78   --nu 1e-2 --rhs generated --tau 0.3 --nev 60 $COARSE
    run tab2_2d_nu1e-2_2L $NP --generate convdiff2d --nx 1200 --nu 1e-2 --rhs generated --tau 0.3 --nev 60 $COARSE
    for m in G3_circuit memchip CoupCons3D nxp1; do
        [ -f $MAT/$m.mtx ] || { printf "\n--- %s: missing, run get_matrices.sh" $m; continue; }
        run tab1_${m}_2L $NP --matrix $MAT/$m.mtx --tau 0.6 --nev 300 --maxit 200 $COARSE
    done
    [ -f $MAT/G3_circuit.mtx ] &&
        run tab1_G3_circuit_2L_t0.9 $NP --matrix $MAT/G3_circuit.mtx --tau 0.9 --nev 300 --maxit 200 $COARSE
fi

# CAL - the same problem with the root and the distributed coarse solver, growing n0:
#   tab2_3d nu=1e-4 (n0 ~ 0.7k), tab2_3d nu=1 (~2.5k), tab2_2d nu=1 (~7.4k),
#   G3_circuit tau = 0.02 ... 0.6 (~0.9k ... 25k), ecology2 / thermal2 tau = 0.6 (~13k / 16k).
if [ $RUN_CAL = 1 ]; then
    for cs in root cp48; do
        [ $cs = root ] && opt="--coarse-solver root" || opt="$COARSE"
        run cal_3d_nu1e-4_$cs $NP --generate convdiff3d --nx 78 --nu 1e-4 --rhs generated --tau 0.3 --nev 60 $opt
        run cal_3d_nu1_$cs    $NP --generate convdiff3d --nx 78 --nu 1    --rhs generated --tau 0.3 --nev 60 $opt
        run cal_2d_nu1_$cs    $NP --generate convdiff2d --nx 1200 --nu 1  --rhs generated --tau 0.3 --nev 60 $opt
        if [ -f $MAT/G3_circuit.mtx ]; then
            for tau in 0.02 0.05 0.1 0.3 0.6; do
                run cal_G3_tau${tau}_$cs $NP --matrix $MAT/G3_circuit.mtx --tau $tau --nev 300 --maxit 200 $opt
            done
        fi
        for m in ecology2 thermal2; do
            [ -f $MAT/$m.mtx ] && run cal_${m}_$cs $NP --matrix $MAT/$m.mtx --tau 0.6 --nev 300 --maxit 200 $opt
        done
    done
    # One line per problem: n0, iterations, A_00 factorization and coarse solve (max over ranks)
    # with both solvers. Per iteration = coarse solve time / iterations (one application per iteration).
    {
        printf "%-16s %7s | %-34s | %-34s\n" "problem" "n0" "root: its  fact[s]  solve/it[ms]" \
               "cp48: its  fact[s]  solve/it[ms]"
        for f in $OUT/log.cal_*_root; do
            b=$(basename $f); b=${b#log.cal_}; b=${b%_root}
            line="$(printf "%-16s %7s" $b "$(awk '$1=="n0"{print $2}' $f)")"
            for cs in root cp48; do
                g=$OUT/log.cal_${b}_$cs
                it=$(grep -oE "in [0-9]+ iterations" $g | grep -oE "[0-9]+")
                fa=$(awk '$1=="setup.coarse_factorization"{print $4}' $g)
                so=$(awk '$1=="solve.preconditioner.coarse.solve"{print $4}' $g)
                line="$line | $(printf "%-4s %8s %10s" "${it:--}" "${fa:--}" \
                       "$( [ -n "$it" ] && [ -n "$so" ] && awk -v s=$so -v i=$it 'BEGIN{printf "%.2f", 1000*s/i}' || echo -)")"
                line="$line          "
            done
            echo "$line"
        done
    } > $OUT/cal_summary.txt
    printf "\n--- root vs cpardiso: %s/cal_summary.txt" $OUT
fi

# THR - finan512 at tau = 0.6: n0 = 109 crashed inside cluster_sparse_solver (run_22843541);
# with the switch the coarse problem goes to rank 0 (the log must show "coarse solver  root(...)").
if [ $RUN_THR = 1 ] && [ -f $MAT/finan512.mtx ]; then
    run tab1_finan512_2L $NP --matrix $MAT/finan512.mtx --tau 0.6 --nev 300 --maxit 200 $COARSE --coarse-root-below $THR
fi

printf "\n--- Done: results in %s\n" $OUT
