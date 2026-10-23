# Functions shared by Allprep_gamg.sh and Allrun_gamg.sh (sourced, OpenFOAM environment loaded).

GAMG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Fingerprint of the mesh template: a prepared case is reused only if it matches.
mesh_md5() { md5sum < $GAMG_DIR/case/system/blockMeshDict | cut -c1-8; }

# factor3 N -> "a b c", a*b*c = N, as close to each other as possible (hierarchical decomposition).
factor3() {
    python3 - $1 <<'EOF'
import sys
n = int(sys.argv[1])
best = None
for a in range(1, n + 1):
    if n % a:
        continue
    for b in range(a, n // a + 1):
        if (n // a) % b or n // a // b < b:
            continue
        c = n // a // b
        if best is None or c / a < best[0]:
            best = (c / a, a, b, c)
print(best[1], best[2], best[3])
EOF
}

# decompose DIR NP METHOD: decomposePar of the mesh of DIR on NP processes.
decompose() {
    local d=$1 np=$2 m=$3
    sed -i "s/^method .*/method          $m;/" $d/system/decomposeParDict
    if [ "$m" = hierarchical ] || [ "$m" = simple ]; then
        sed -i "s/^\( *n  *\)(.*);/\1($(factor3 $np));/" $d/system/decomposeParDict
    fi
    rm -rf $d/processor*
    decomposePar -case $d -force > $d/log.decomposePar.$m 2>&1 &&
        [ "$(ls -d $d/processor* 2>/dev/null | wc -l)" = "$np" ] && echo $m > $d/METHOD
}

# prep_case N NP DIR METHOD: case of the template with an N^3 mesh, decomposed on NP processes
# (NP = 1: serial case). Writes DIR/PREPARED when done.
prep_case() {
    local n=$1 np=$2 d=$3 method=${4:-scotch}
    rm -rf $d && mkdir -p $d && cp -r $GAMG_DIR/case/0 $GAMG_DIR/case/system $d/ || return 1
    sed -i -e "s/^nx .*/nx      $n;/" -e "s/^ny .*/ny      $n;/" -e "s/^nz .*/nz      $n;/" $d/system/gridSize
    sed -i "s/^numberOfSubdomains .*/numberOfSubdomains $np;/" $d/system/decomposeParDict
    local t0; t0=$(date +%s)
    blockMesh -case $d > $d/log.blockMesh 2>&1 || { echo "blockMesh failed: $d/log.blockMesh"; return 1; }
    echo serial > $d/METHOD
    if [ "$np" -gt 1 ]; then
        if ! decompose $d $np $method; then
            [ "$method" != hierarchical ] && echo "decomposePar ($method) failed in $d: hierarchical"
            decompose $d $np hierarchical || { echo "decomposePar failed: $d/log.decomposePar.*"; return 1; }
        fi
        rm -rf $d/constant/polyMesh   # only the decomposed mesh is used
    fi
    echo "N=$n NP=$np method=$(cat $d/METHOD) mesh=$(mesh_md5) time=$(( $(date +%s) - t0 ))s $(date '+%F %T')" > $d/PREPARED
}

# case_ok DIR N NP: prepared case with the right size, processes and mesh template.
case_ok() {
    [ -f $1/PREPARED ] && grep -q "N=$2 NP=$3 .*mesh=$(mesh_md5) " $1/PREPARED
}

# set_dicts DIR: solver controls, schemes and bubbles of the current template, then the bubble
# parameters of the campaign (NBUB RADIUS RHO CFL SEED STEPS CORR).
set_dicts() {
    local d=$1 f
    for f in controlDict fvSchemes fvSolution bubblesDict; do cp $GAMG_DIR/case/system/$f $d/system/$f; done
    sed -i -e "s/^nBubbles .*/nBubbles    $NBUB;/" -e "s/^radius .*/radius      $RADIUS;/" \
           -e "s/^rhoRatio .*/rhoRatio    $RHO;/" -e "s/^cfl .*/cfl         $CFL;/" \
           -e "s/^seed .*/seed        $SEED;/" -e "s/^steps .*/steps       ${STEPS:-1};/" \
           -e "s/^correctors .*/correctors  $CORR;/" $d/system/bubblesDict
}

# build_app DIR: bubblyPressureFoam built in DIR (copy of the sources), executable DIR/bin/...
# The bubble field (bubbleFieldCapi.cpp) is compiled with the schwarz2lvl compiler and flags
# (S2L_CXX, S2L_CXXFLAGS), the rest with wmake (OpenFOAM toolchain).
build_app() {
    local d=$1 repo; repo=$(cd $GAMG_DIR/../.. && pwd)
    rm -rf $d/app && mkdir -p $d/bin && cp -r $repo/openfoam/bubblyPressureFoam $d/app || return 1
    (
        cd $d/app &&
        $S2L_CXX $S2L_CXXFLAGS -fPIC -I$repo/include/utils -c bubbleFieldCapi.cpp -o bubbleFieldCapi.o &&
        S2L_BUBBLE_OBJ=$d/app/bubbleFieldCapi.o FOAM_USER_APPBIN=$d/bin wmake
    ) && [ -x $d/bin/bubblyPressureFoam ]
}

# matrix_check DIR APP MATGEN: 24^3 serial case, OpenFOAM matrix of step 3 (-writeMatrix) against
# the generator (matgen). Last line of DIR/log.check_matrix: OK / DIFFERENT. Returns 0 if OK.
matrix_check() {
    local d=$1 app=$2 matgen=$3
    prep_case 24 1 $d/check24 > $d/log.check_prep 2>&1 && STEPS=4 set_dicts $d/check24 &&
        ( timeout 300 $app -case $d/check24 -solver GAMG_GS -steps 4 -correctors 1 -writeMatrix $d/check_of \
              -csv $d/check_of.csv > $d/log.check_app 2>&1 ) &&
        ( cd $d && timeout 300 $matgen bubbles3d 24 nb=$NBUB radius=$RADIUS rho=$RHO cfl=$CFL t=3 seed=$SEED \
              out=$d/check_s2l > $d/log.check_matgen 2>&1 ) &&
        python3 $GAMG_DIR/compare_mtx.py $d/check_of $d/check_s2l > $d/log.check_matrix 2>&1
}
