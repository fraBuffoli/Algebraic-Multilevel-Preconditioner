#!/bin/bash
#SBATCH --nodes=1                    	# one whole node: 9 decompositions side by side need ~250 GB
#SBATCH --ntasks-per-node=48
#SBATCH --exclusive
#SBATCH --partition=g100_usr_prod    	# server type
#SBATCH --error=allprep_scaling.err  	# standard error file
#SBATCH --output=allprep_scaling.out 	# standard output file
#SBATCH --account=IscrC_EXTA-PMF       	# account name
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=francesco.buffoli@mail.polimi.it
#SBATCH --job-name=s2l_scal_prep
#SBATCH --time=01:00:00
# -------------------------------------------------------------------------------- #
# Preparation of the scaling campaign (Allrun_scaling.sh), on one node:
#  - one serial mesh per size, cases/base<N> (blockMesh, unit cube N^3), then one decomposed case
#    per size and process count, cases/bub<N>_np<NP> (decomposePar, scotch; hierarchical as a
#    fallback), made from the linked base mesh. Cases already prepared with the same template are
#    kept (also those of Allprep_gamg.sh: same names).
#  - meanwhile: schwarz2lvl, the unit tests, matgen and bubblyPressureFoam built once in
#    cases/bin_<sources md5>, then test_bubbles and the matrix check (OpenFOAM assembles the generated
#    matrix, 24^3). Allrun_scaling.sh uses these binaries if the sources have not changed since.
# If anything fails the job fails, and the dependent Allrun_scaling.sh jobs stay pending (scancel them).
#
#   from test/gamg:   ./submit_scaling.sh        (this job + one Allrun_scaling.sh per process count)
# Disk: ~25 GB in cases/ for 192^3, 256^3 and 320^3 on 240, 336 and 432 processes; delete when done.
# Local smoke test (OpenFOAM-10 environment sourced):  SMOKE=1 NPS="2 4" ./Allprep_scaling.sh
# -------------------------------------------------------------------------------- #
SIZES=${SIZES:-"192 256 320"}    # cells per direction
NPS=${NPS:-"240 336 432"}        # process counts of the Allrun_scaling.sh jobs
METHOD=${METHOD:-scotch}         # decomposition method (hierarchical as fallback)
NBUB=150; RADIUS=0.04; RHO=1000; CFL=0.5; SEED=2024; CORR=1   # bubbles of the campaign (matrix check)
S2L_CXX=${S2L_CXX:-mpicxx}
S2L_CXXFLAGS=${S2L_CXXFLAGS:-"-std=c++17 -O3 -march=skylake-avx512 -DNDEBUG"}   # as in Allrun_scaling.sh

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
if [ "${SMOKE:-0}" = 1 ]; then
    module() { :; }
    S2L_CXXFLAGS="-std=c++17 -O2 -DNDEBUG"
    SIZES=${SMOKE_SIZES:-"16 20"}
    MPIRUN_OPTS=${MPIRUN_OPTS:-"--oversubscribe"}
    CASES=${CASES:-$PWD/cases_smoke}
    METHOD=hierarchical
fi
module load profile/eng
module load autoload openfoam/10
module load metis/5.1.0--gcc--10.2.0
module load intel-oneapi-mkl/2023.2.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset I_MPI_PMI_LIBRARY
[ -n "$WM_PROJECT_DIR" ] || { echo "OpenFOAM environment not loaded"; exit 1; }
CASES=${CASES:-$PWD/cases}
mkdir -p $CASES
. ./gamg_functions.sh
T0=$(date +%s)

# Meshes: one chain per size (blockMesh, then the decompositions of that size side by side).
for n in $SIZES; do
    (
        missing=""
        for np in $NPS; do case_ok $CASES/bub${n}_np$np $n $np || missing="$missing $np"; done
        [ -z "$missing" ] && exit 0
        b=$CASES/base$n
        if ! case_ok $b $n 1; then
            prep_case $n 1 $b > $CASES/log.base$n 2>&1 || { echo "base$n: FAILED, see $CASES/log.base$n"; exit 1; }
        fi
        echo "base$n: $(cat $b/PREPARED)"
        for np in $missing; do prep_case $n $np $CASES/bub${n}_np$np $METHOD $b > $CASES/log.bub${n}_np$np 2>&1 & done
        wait
    ) &
done

# Meanwhile: binaries and checks (once per version of the sources).
B=$CASES/bin_$(src_md5)
if [ -f $B/OK ]; then
    echo "binaries: $B already built and checked ($(cat $B/OK))"
    fail=0
else
    rm -rf $B && mkdir -p $B
    MAKE_JOBS=${MAKE_JOBS:-36}
    if build_and_check $B > $B/log.build_and_check 2>&1; then
        echo "binaries: $B ($(cat $B/OK))"
        fail=0
    else
        echo "binaries / checks: FAILED: $(cat $B/log.build_and_check)"
        fail=1
    fi
fi
wait

for n in $SIZES; do
    for np in $NPS; do
        d=$CASES/bub${n}_np$np
        if case_ok $d $n $np; then echo "$d: $(cat $d/PREPARED)"; else echo "$d: FAILED, see $CASES/log.bub${n}_np$np, $d/log.*"; fail=1; fi
    done
done
echo "prep: $(( $(date +%s) - T0 )) s, cases/: $(du -sh $CASES 2>/dev/null | cut -f1)"
exit $fail
