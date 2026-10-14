#!/bin/bash
#SBATCH --nodes=5                    	# 5 nodes
#SBATCH --ntasks-per-node=48         	# 48 tasks per node -> max 240 MPI processes
#SBATCH --exclusive                  	# whole nodes: clean timings
#SBATCH --partition=g100_usr_prod    	# server type
#SBATCH --error=allrun.err            	# standard error file
#SBATCH --output=allrun.out           	# standard output file
#SBATCH --account=IscrC_EXTA-PMF       	# account name
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=francesco.buffoli@mail.polimi.it
#SBATCH --job-name=s2l_paper
#SBATCH --time=06:00:00
# -------------------------------------------------------------------------------- #
# Reproduction of the experiments of Section 4 of
#   Al Daas, Jolivet, Rees, "Efficient algebraic two-level Schwarz preconditioner
#   for sparse matrices", SIAM J. Sci. Comput. 45 (2023)
# within 5 nodes (one subdomain per MPI process).
#
#   1. once, on a LOGIN node (compute nodes are offline):  ./get_matrices.sh
#   2. from test/paper:                                     sbatch Allrun.sh
#   3. tables and plots:                                    python3 plot_paper.py run_<jobid>
# -------------------------------------------------------------------------------- #

## what to run (1 = on), in this order (cheapest first)
RUN_CHECK=1          # build + MPI unit tests on 16 processes                       ~ 3 min
RUN_FIG4=1           # strong scaling, 3D convection-diffusion, nu = 1e-2 (Fig. 4)  ~ 15 min
RUN_TAB2=1           # 2D/3D convection-diffusion, nu = 1 ... 1e-4 (Table 2)        ~ 15 min
RUN_TAB1=1           # SuiteSparse matrices, tau = 0.6, nev = 300 (Table 1)         ~ 1-3 h
RUN_FIG2=1           # G3_circuit, sweep on tau (Fig. 2)                            ~ 0.5-1 h

NP=240               # processes = subdomains for Tables 1-2 and Fig. 2 (paper: 256)
RUN_TIMEOUT=1200     # seconds: a run that does not finish is stopped, the job goes on
MATRICES="finan512 nxp1 CoupCons3D ecology2 tmt_unsym thermal2 atmosmodj G3_circuit Transport memchip circuit5M_dc"

# -------------------------------------------------------------------------------- #
# environment: the OpenFOAM 10 one (compiler + Intel MPI) + METIS + MKL (PARDISO).
# The MKL module without MPI tag does not replace OpenFOAM's Intel MPI.
module load profile/eng
module load autoload openfoam/10
module load metis/5.1.0--gcc--10.2.0
module load intel-oneapi-mkl/2023.2.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # one sequential LU per process

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
REPO=$(cd ../.. && pwd)
OUT=$PWD/run_${SLURM_JOB_ID:-local}
BIN=$REPO/bin/schwarz2lvl
MAT=$REPO/matrices
mkdir -p $OUT/csv $OUT/hist
module list > $OUT/log.modules 2>&1
git -C $REPO log -1 --format='%h %s' > $OUT/log.git 2>&1

# -------------------------------------------------------------------------------- #
# run NAME NP <schwarz2lvl options>  ->  log.NAME, csv/NAME.csv, hist/NAME.txt, status.txt
run() {
    local name=$1 np=$2 t0 st; t0=$(date +%s); shift 2
    printf "\n--- %-26s np=%-4s " "$name" "$np"
    timeout $RUN_TIMEOUT mpirun -n $np $BIN "$@" \
        --csv $OUT/csv/$name.csv --history $OUT/hist/$name.txt > $OUT/log.$name 2>&1
    case $? in 0) st=ok ;; 2) st=not_converged ;; 124) st=timeout ;; *) st=error ;; esac
    printf "%s (%ss)" $st $(( $(date +%s) - t0 ))
    echo "$name $np $st $(( $(date +%s) - t0 ))" >> $OUT/status.txt
}

# -------------------------------------------------------------------------------- #
printf "\n--- Building ---"
libdir() { local d; for d in ${LD_LIBRARY_PATH//:/ }; do ls $d/$1.so* > /dev/null 2>&1 && { echo $d; return; }; done; }
METIS_DIR=$(libdir libmetis)                          # .../metis-5.1.0-xxx/lib
MKL_DIR=${MKLROOT:+$MKLROOT/lib/intel64}; MKL_DIR=${MKL_DIR:-$(libdir libmkl_core)}   # .../mkl/2023.2.0/lib/intel64
( cd $REPO && make clean > /dev/null && make -j 48 all tests \
    CXX=mpicxx EIGEN_INC=external/eigen3 CXXFLAGS="-std=c++17 -O3 -march=skylake-avx512 -DNDEBUG" \
    METIS_INC="$(dirname "$METIS_DIR")/include" METIS_LIB="-L$METIS_DIR -Wl,-rpath,$METIS_DIR -lmetis" \
    USE_MUMPS=0 USE_UMFPACK=0 USE_PARDISO=1 MKL_INC="$(dirname "$(dirname "$MKL_DIR")")/include" \
    MKL_LIB="-L$MKL_DIR -Wl,-rpath,$MKL_DIR -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl" \
  ) > $OUT/log.build 2>&1 || { printf "\nbuild FAILED: see %s/log.build\n" $OUT; exit 1; }

if [ $RUN_CHECK = 1 ]; then
    printf "\n--- Unit tests ---"
    mpirun -n 16 $REPO/bin/test_subdomain      > $OUT/log.test_subdomain 2>&1
    mpirun -n 16 $REPO/bin/test_preconditioner > $OUT/log.test_preconditioner 2>&1
    printf "\n    test_subdomain: %s   test_preconditioner: %s" \
        "$(tail -n1 $OUT/log.test_subdomain)" "$(tail -n1 $OUT/log.test_preconditioner)"
fi

# Fig. 4 - strong scaling of (4.1) in 3D, nu = 1e-2, tau = 0.3, nev = 60.
# Paper: n = 8.1e6 (P2), N = 32 ... 512. Here: n = 200^3 = 8.0e6 (finite volumes), N = 30 ... 240.
if [ $RUN_FIG4 = 1 ]; then
    for N in 30 60 120 240; do
        run fig4_N$N $N --generate convdiff3d --nx 200 --nu 1e-2 --rhs generated --tau 0.3 --nev 60
    done
fi

# Table 2 - nu = 1 ... 1e-4, tau = 0.3, nev = 60, with the subdomain size of the paper
# (~6k unknowns per subdomain in 2D, ~2k in 3D): 1200^2 and 78^3 unknowns on NP subdomains.
if [ $RUN_TAB2 = 1 ]; then
    for nu in 1 1e-1 1e-2 1e-3 1e-4; do
        run tab2_2d_nu${nu}_2L  $NP --generate convdiff2d --nx 1200 --nu $nu --rhs generated --tau 0.3 --nev 60
        run tab2_2d_nu${nu}_RAS $NP --generate convdiff2d --nx 1200 --nu $nu --rhs generated --coarse none
        run tab2_3d_nu${nu}_2L  $NP --generate convdiff3d --nx 78   --nu $nu --rhs generated --tau 0.3 --nev 60
        run tab2_3d_nu${nu}_RAS $NP --generate convdiff3d --nx 78   --nu $nu --rhs generated --coarse none
    done
fi

# Table 1 - RAS + deflated coarse correction (2.4), tau = 0.6, nev = 300, random rhs,
# GMRES(30), rtol 1e-8 (the paper reports > 100 iterations as a failure). RAS = one-level reference.
if [ $RUN_TAB1 = 1 ]; then
    for m in $MATRICES; do
        [ -f $MAT/$m.mtx ] || { printf "\n--- %s: missing, run get_matrices.sh" $m; continue; }
        run tab1_${m}_RAS $NP --matrix $MAT/$m.mtx --coarse none --maxit 200
        run tab1_${m}_2L  $NP --matrix $MAT/$m.mtx --tau 0.6 --nev 300 --maxit 200
    done
fi

# Fig. 2 - G3_circuit, tau = 0.02 ... 0.9 (nev = 300), and the one-level ASM of the figure.
if [ $RUN_FIG2 = 1 ] && [ -f $MAT/G3_circuit.mtx ]; then
    run fig2_ASM $NP --matrix $MAT/G3_circuit.mtx --one-level asm --coarse none --maxit 200
    for tau in 0.02 0.05 0.1 0.3 0.6 0.9; do
        [ $tau = 0.6 ] && [ $RUN_TAB1 = 1 ] && continue     # = tab1_G3_circuit_2L
        run fig2_tau$tau $NP --matrix $MAT/G3_circuit.mtx --tau $tau --nev 300 --maxit 200
    done
fi

printf "\n--- Done: results in %s\n" $OUT
