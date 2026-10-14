#!/bin/bash
#SBATCH --nodes=5                    	# 5 nodes
#SBATCH --ntasks-per-node=48         	# 48 tasks per node -> max 240 MPI processes
#SBATCH --exclusive                  	# whole nodes: clean timings
#SBATCH --partition=g100_usr_prod    	# server type
#SBATCH --error=allrun_coarse.err      	# standard error file
#SBATCH --output=allrun_coarse.out     	# standard output file
#SBATCH --account=IscrC_EXTA-PMF       	# account name
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=francesco.buffoli@mail.polimi.it
#SBATCH --job-name=s2l_coarse
#SBATCH --time=06:00:00
# -------------------------------------------------------------------------------- #
# Second campaign: distributed factorization of the coarse matrix A_00 and
# multiplicity partition of unity, on the experiments of Allrun.sh.
#   - coarse solvers: "root" (A_00 gathered and factorized on rank 0, as in Allrun.sh)
#     vs "cpardiso" (MKL cluster_sparse_solver, distributed over K processes)
#   - partitions of unity: boolean (as in Allrun.sh) vs multiplicity
#
#   1. matrices already in ../../matrices (get_matrices.sh)
#   2. from test/paper:                       sbatch Allrun_coarse.sh
#   3. tables and plots:                      python3 plot_coarse.py run_coarse_<jobid>
# The binary is built inside the run directory: it does not touch ../../bin, so this job
# can run while Allrun.sh is running.
# -------------------------------------------------------------------------------- #

## what to run (1 = on), in this order
RUN_CHECK=1          # unit tests (all coarse solvers) + root vs cpardiso on 240 processes  ~ 5 min
RUN_EIG=1            # Table 1 matrices: adaptive k (default) vs nev = 300 computed         ~ 15 min
RUN_TAB1=1           # SuiteSparse matrices: coarse solvers at tau = 0.6 and 0.9, PoU      ~ 1-1.5 h
RUN_FIG2=1           # G3_circuit, sweep on tau, both PoU                                  ~ 15 min
RUN_TAB2=1           # convection-diffusion, nu = 1 ... 1e-4, both PoU (two-level only)    ~ 15 min
RUN_FIG4=1           # strong scaling 3D, nu = 1e-2, both PoU                               ~ 25 min
RUN_DIAG=1           # circuit5M_dc with per-rank progress of the eigensolver               < 30 min

NP=240               # processes = subdomains
RUN_TIMEOUT=1200     # seconds: a run that does not finish is stopped, the job goes on
POUS="boolean multiplicity"
COARSE="root cp240 cp48 cp10"   # cpK = cpardiso on K processes (one leader every 240/K ranks)
CP=cp48                         # coarse solver of the multiplicity pass of TAB1 and of FIG2 / TAB2 / FIG4
# circuit5M_dc is not in the list: its two-level setup did not end in 20 min (run 22838900), see RUN_DIAG
MATRICES="finan512 nxp1 CoupCons3D ecology2 tmt_unsym thermal2 atmosmodj G3_circuit Transport memchip"

# -------------------------------------------------------------------------------- #
# environment: the OpenFOAM 10 one (compiler + Intel MPI) + METIS + MKL (PARDISO).
module load profile/eng
module load autoload openfoam/10
module load metis/5.1.0--gcc--10.2.0
module load intel-oneapi-mkl/2023.2.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # one sequential LU per process

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
REPO=$(cd ../.. && pwd)
OUT=$PWD/run_coarse_${SLURM_JOB_ID:-local}
BIN=$OUT/bin/schwarz2lvl
MAT=$REPO/matrices
mkdir -p $OUT/csv $OUT/hist
module list > $OUT/log.modules 2>&1
git -C $REPO log -1 --format='%h %s' > $OUT/log.git 2>&1
echo "NP=$NP POUS='$POUS' COARSE='$COARSE' CP=$CP" > $OUT/config.txt

# -------------------------------------------------------------------------------- #
# run NAME NP <schwarz2lvl options>  ->  log.NAME, csv/NAME.csv, hist/NAME.txt, status.txt
run() {
    local name=$1 np=$2 t0 st rc; t0=$(date +%s); shift 2
    printf "\n--- %-34s np=%-4s " "$name" "$np"
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
# coarse solver tag -> options:  root | cpK
copts() { case $1 in root) echo "--coarse-solver root" ;; cp*) echo "--coarse-solver cpardiso --coarse-procs ${1#cp}" ;; esac; }
pt() { echo ${1:0:4}; }         # boolean -> bool, multiplicity -> mult (run names)

# -------------------------------------------------------------------------------- #
printf "\n--- Building (USE_CPARDISO=1) ---"
libdir() { local d; for d in ${LD_LIBRARY_PATH//:/ }; do ls $d/$1.so* > /dev/null 2>&1 && { echo $d; return; }; done; }
METIS_DIR=$(libdir libmetis)
MKL_DIR=${MKLROOT:+$MKLROOT/lib/intel64}; MKL_DIR=${MKL_DIR:-$(libdir libmkl_core)}
ls $MKL_DIR/libmkl_blacs_intelmpi_lp64.so* > /dev/null 2>&1 ||
    { printf "\nMKL BLACS for Intel MPI not found in %s: cluster_sparse_solver unavailable\n" "$MKL_DIR"; exit 1; }
( cd $REPO && make -j 48 all tests BUILD=$OUT/build BIN=$OUT/bin \
    CXX=mpicxx EIGEN_INC=external/eigen3 CXXFLAGS="-std=c++17 -O3 -march=skylake-avx512 -DNDEBUG" \
    METIS_INC="$(dirname "$METIS_DIR")/include" METIS_LIB="-L$METIS_DIR -Wl,-rpath,$METIS_DIR -lmetis" \
    USE_MUMPS=0 USE_UMFPACK=0 USE_PARDISO=1 USE_CPARDISO=1 MKL_INC="$(dirname "$(dirname "$MKL_DIR")")/include" \
    MKL_LIB="-L$MKL_DIR -Wl,-rpath,$MKL_DIR -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl" \
    MKL_BLACS_LIB="-lmkl_blacs_intelmpi_lp64" \
  ) > $OUT/log.build 2>&1 || { printf "\nbuild FAILED: see %s/log.build\n" $OUT; exit 1; }

# Unit tests (every coarse solver vs a dense reference) and a short root vs cpardiso
# comparison on all the processes, symmetric (mtype -2) and nonsymmetric (mtype 11):
# same n0 and iterations expected. If cpardiso fails here, the job stops.
if [ $RUN_CHECK = 1 ]; then
    printf "\n--- Unit tests ---"
    mpirun -n 16 $OUT/bin/test_subdomain      > $OUT/log.test_subdomain 2>&1
    mpirun -n 16 $OUT/bin/test_preconditioner > $OUT/log.test_preconditioner 2>&1
    printf "\n    test_subdomain: %s   test_preconditioner: %s (%s cpardiso checks)" \
        "$(tail -n1 $OUT/log.test_subdomain)" "$(tail -n1 $OUT/log.test_preconditioner)" \
        "$(grep -c 'OK.*cpardiso' $OUT/log.test_preconditioner)"
    for g in "laplace3d --nx 100" "convdiff3d --nx 100 --nu 1e-2"; do
        for pou in $POUS; do
            for c in $COARSE; do
                run check_${g%% *}_$(pt $pou)_$c $NP --generate $g --tau 0.3 --nev 60 --pou $pou $(copts $c)
            done
        done
    done
    if grep -E "^check_.*_cp[0-9]+ " $OUT/status.txt | grep -qv " ok "; then
        printf "\n\ncpardiso failed in the checks: campaign stopped (see %s/log.check_*)\n" $OUT; exit 1
    fi
fi

# Local eigensolver: by default k = 32, 64, ... eigenpairs are computed until the smallest is below
# 1/tau (--nev-start 32), instead of always computing nev = 300 and discarding the rest as in
# run 22838900. Same selected vectors, so n0 and iterations must be equal; the runs below with
# the old behaviour (--nev-start 0) are the reference for tab1_<matrix>_bool_t0.6_root.
if [ $RUN_EIG = 1 ]; then
    for m in $MATRICES; do
        [ -f $MAT/$m.mtx ] || continue
        run eig_${m}_nevfixed $NP --matrix $MAT/$m.mtx --tau 0.6 --nev 300 --nev-start 0 --maxit 200 $(copts root)
    done
fi

# Table 1 - RAS + deflated coarse correction (2.4), nev = 300, random rhs, GMRES(30), rtol 1e-8.
# tau = 0.6 is the setting of the paper; but its G3_circuit row (n0 = 71,385, 8 it.) is the tau = 0.9
# point of its own Fig. 2, and our tau = 0.9 gives the same n0/N (278.8): the tau = 0.9 pass tells
# whether the n0 of Table 1 are generally tau = 0.9 ones, and loads the coarse solvers with a 2-3x
# larger A_00. The one-level RAS depends on D_i, so it is run for each PoU.
tab1_pass() {   # PoU tau "coarse solvers" with-RAS(0/1)
    local pou=$1 tau=$2 cs=$3 ras=$4 m c
    for m in $MATRICES; do
        [ -f $MAT/$m.mtx ] || { printf "\n--- %s: missing, run get_matrices.sh" $m; continue; }
        [ $ras = 1 ] && run tab1_${m}_$(pt $pou)_RAS $NP --matrix $MAT/$m.mtx --pou $pou --coarse none --maxit 200
        for c in $cs; do
            run tab1_${m}_$(pt $pou)_t${tau}_$c $NP --matrix $MAT/$m.mtx --pou $pou --tau $tau --nev 300 --maxit 200 $(copts $c)
        done
    done
}
if [ $RUN_TAB1 = 1 ]; then
    tab1_pass boolean      0.6 "$COARSE" 1     # every coarse solver, paper setting
    tab1_pass boolean      0.9 "$COARSE" 0     # every coarse solver, larger A_00; n0 vs paper
    tab1_pass multiplicity 0.6 "$CP"     1     # PoU: n0 and iterations do not depend on the solver
fi

# Fig. 2 - G3_circuit, tau = 0.02 ... 0.9 (nev = 300), both PoU.
if [ $RUN_FIG2 = 1 ] && [ -f $MAT/G3_circuit.mtx ]; then
    for pou in $POUS; do
        for tau in 0.02 0.05 0.1 0.3 0.6 0.9; do
            run fig2_$(pt $pou)_tau$tau $NP --matrix $MAT/G3_circuit.mtx --pou $pou --tau $tau --nev 300 --maxit 200 $(copts $CP)
        done
    done
fi

# Table 2 - nu = 1 ... 1e-4, tau = 0.3, nev = 60, paper subdomain size (1200^2 and 78^3 on NP).
if [ $RUN_TAB2 = 1 ]; then
    for pou in $POUS; do
        for nu in 1 1e-1 1e-2 1e-3 1e-4; do
            run tab2_2d_nu${nu}_$(pt $pou) $NP --generate convdiff2d --nx 1200 --nu $nu --rhs generated --pou $pou --tau 0.3 --nev 60 $(copts $CP)
            run tab2_3d_nu${nu}_$(pt $pou) $NP --generate convdiff3d --nx 78   --nu $nu --rhs generated --pou $pou --tau 0.3 --nev 60 $(copts $CP)
        done
    done
fi

# Fig. 4 - strong scaling of (4.1) in 3D, nu = 1e-2, tau = 0.3, nev = 60, n = 200^3.
if [ $RUN_FIG4 = 1 ]; then
    for pou in $POUS; do
        for N in 30 60 120 240; do
            run fig4_$(pt $pou)_N$N $N --generate convdiff3d --nx 200 --nu 1e-2 --rhs generated --pou $pou --tau 0.3 --nev 60 $(copts $CP)
        done
    done
fi

# circuit5M_dc: in run 22838900 the two-level setup was still running after 20 min. --verbose 2
# prints one line per rank (stderr, in the log) when its eigensolver ends: the missing ranks are the
# slow ones, and the lines show n_i, n_Gamma, ker(A~), operator applications of the others.
if [ $RUN_DIAG = 1 ] && [ -f $MAT/circuit5M_dc.mtx ]; then
    RUN_TIMEOUT=1800 run diag_circuit5M_dc_2L $NP --matrix $MAT/circuit5M_dc.mtx --tau 0.6 --nev 300 --maxit 200 \
        --verbose 2 $(copts $CP)
fi

printf "\n--- Done: results in %s\n" $OUT
