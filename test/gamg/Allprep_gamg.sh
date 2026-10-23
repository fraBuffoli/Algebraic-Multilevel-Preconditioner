#!/bin/bash
#SBATCH --nodes=1                    	# part of one node: serial OpenFOAM tools
#SBATCH --ntasks=4                   	# blockMesh / decomposePar of the sizes run side by side
#SBATCH --mem=120G                   	# decomposePar of 256^3 needs ~20 GB
#SBATCH --partition=g100_usr_prod    	# server type
#SBATCH --error=allprep_gamg.err     	# standard error file
#SBATCH --output=allprep_gamg.out    	# standard output file
#SBATCH --account=IscrC_EXTA-PMF       	# account name
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=francesco.buffoli@mail.polimi.it
#SBATCH --job-name=s2l_gamg_prep
#SBATCH --time=00:40:00
# -------------------------------------------------------------------------------- #
# Meshes and checks of the GAMG reference (Allrun_gamg.sh), on part of one node:
#  - unit cube N^3 (blockMesh) decomposed on NP processes (decomposePar, scotch), one case per
#    size in cases/bub<N>_np<NP>, kept if already prepared. blockMesh and decomposePar are serial:
#    here they do not keep the 240 cores of the main job idle (~10 minutes).
#  - builds bubblyPressureFoam (bubble field compiled with the schwarz2lvl compiler) and matgen,
#    and checks on a 24^3 case that OpenFOAM assembles the generated matrix (log.check_matrix).
# If anything fails, the job fails and a dependent Allrun_gamg.sh does not start (scancel it).
#
#   1. from test/gamg:   jid=$(sbatch --parsable Allprep_gamg.sh)            (~2 min; ~10-15 min with new meshes)
#   2. then:             sbatch --dependency=afterok:$jid Allrun_gamg.sh
# Delete cases/ when done (~3 GB for 192^3 + 256^3 at 240 processes).
# -------------------------------------------------------------------------------- #
SIZES=${SIZES:-"192 256"}        # cells per direction
NP=${NP:-240}                    # processes of Allrun_gamg.sh
METHOD=${METHOD:-scotch}         # decomposition method (hierarchical as fallback)
NBUB=150; RADIUS=0.04; RHO=1000; CFL=0.5; SEED=2024; CORR=1   # bubbles of the campaign (matrix check)
S2L_CXX=${S2L_CXX:-mpicxx}
S2L_CXXFLAGS=${S2L_CXXFLAGS:-"-std=c++17 -O3 -march=skylake-avx512 -DNDEBUG"}   # as in Allrun_gamg.sh

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
if [ "${SMOKE:-0}" != 1 ]; then
    module load profile/eng
    module load autoload openfoam/10
else
    S2L_CXXFLAGS="-std=c++17 -O2 -DNDEBUG"
fi
[ -n "$WM_PROJECT_DIR" ] || { echo "OpenFOAM environment not loaded"; exit 1; }
CASES=${CASES:-$PWD/cases}
mkdir -p $CASES
. ./gamg_functions.sh

for n in $SIZES; do
    d=$CASES/bub${n}_np${NP}
    if case_ok $d $n $NP; then echo "$d: already prepared"; else prep_case $n $NP $d $METHOD & fi
done

# Meanwhile: application, matgen and the matrix check.
REPO=$(cd ../.. && pwd)
CHECK=$CASES/app_check
rm -rf $CHECK && mkdir -p $CHECK
fail=0
if build_app $CHECK > $CHECK/log.build_app 2>&1 &&
   ( cd $REPO && make -j 4 $CHECK/bin/matgen BUILD=$CHECK/build BIN=$CHECK/bin CXX="$S2L_CXX" \
         CXXFLAGS="$S2L_CXXFLAGS" EIGEN_INC=external/eigen3 USE_MUMPS=0 USE_UMFPACK=0 USE_PARDISO=0 \
         USE_CPARDISO=0 USE_AMGCL=0 ) > $CHECK/log.build_matgen 2>&1 &&
   matrix_check $CHECK $CHECK/bin/bubblyPressureFoam $CHECK/bin/matgen; then
    echo "bubblyPressureFoam: $(tail -n1 $CHECK/log.check_matrix)"
else
    echo "bubblyPressureFoam / matrix check: FAILED, see $CHECK/log.* ($(tail -n1 $CHECK/log.check_matrix 2>/dev/null))"
    fail=1
fi
wait
for n in $SIZES; do
    d=$CASES/bub${n}_np${NP}
    if case_ok $d $n $NP; then echo "$d: $(cat $d/PREPARED)"; else echo "$d: FAILED, see $d/log.*"; fail=1; fi
done
exit $fail
