# Functions shared by Allprep_gamg.sh, Allrun_gamg.sh, Allprep_scaling.sh and Allrun_scaling.sh
# (sourced, OpenFOAM environment loaded).

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

# prep_case N NP DIR METHOD [BASE]: case of the template with an N^3 mesh, decomposed on NP processes
# (NP = 1: serial case). The mesh is made by blockMesh, or taken from the serial case BASE (absolute
# path; its files are linked, not copied, and BASE is not modified). Writes DIR/PREPARED when done.
prep_case() {
    local n=$1 np=$2 d=$3 method=${4:-scotch} base=$5
    rm -rf $d && mkdir -p $d && cp -r $GAMG_DIR/case/0 $GAMG_DIR/case/system $d/ || return 1
    sed -i -e "s/^nx .*/nx      $n;/" -e "s/^ny .*/ny      $n;/" -e "s/^nz .*/nz      $n;/" $d/system/gridSize
    sed -i "s/^numberOfSubdomains .*/numberOfSubdomains $np;/" $d/system/decomposeParDict
    local t0; t0=$(date +%s)
    if [ -n "$base" ]; then
        mkdir -p $d/constant/polyMesh && ln -s $base/constant/polyMesh/* $d/constant/polyMesh/ ||
            { echo "no mesh in $base"; return 1; }
    else
        blockMesh -case $d > $d/log.blockMesh 2>&1 || { echo "blockMesh failed: $d/log.blockMesh"; return 1; }
    fi
    echo serial > $d/METHOD
    if [ "$np" -gt 1 ]; then
        if ! decompose $d $np $method; then
            [ "$method" != hierarchical ] && echo "decomposePar ($method) failed in $d: hierarchical"
            decompose $d $np hierarchical || { echo "decomposePar failed: $d/log.decomposePar.*"; return 1; }
        fi
        rm -rf $d/constant/polyMesh   # only the decomposed mesh is used (links only, if from BASE)
    fi
    echo "N=$n NP=$np method=$(cat $d/METHOD) mesh=$(mesh_md5) time=$(( $(date +%s) - t0 ))s $(date '+%F %T')" > $d/PREPARED
}

# case_ok DIR N NP: prepared case with the right size, processes and mesh template.
case_ok() {
    [ -f $1/PREPARED ] && grep -q "N=$2 NP=$3 .*mesh=$(mesh_md5) " $1/PREPARED
}

# set_dicts DIR: solver controls, schemes and bubbles of the current template, then the bubble
# parameters of the campaign (NBUB RADIUS RHO CFL SEED STEPS CORR [RELTOL] [SOURCE] [W_RTOL W_ATOL]).
# SOURCE: right-hand side, alpha (default) or buoyancy. W_RTOL / W_ATOL: relTol / tolerance of the
# W_* variants of the warm-start campaign (template: 0.05 / 1e-7).
set_dicts() {
    local d=$1 f
    for f in controlDict fvSchemes fvSolution bubblesDict; do cp $GAMG_DIR/case/system/$f $d/system/$f; done
    sed -i -e "s/^nBubbles .*/nBubbles    $NBUB;/" -e "s/^radius .*/radius      $RADIUS;/" \
           -e "s/^rhoRatio .*/rhoRatio    $RHO;/" -e "s/^cfl .*/cfl         $CFL;/" \
           -e "s/^seed .*/seed        $SEED;/" -e "s/^steps .*/steps       ${STEPS:-1};/" \
           -e "s/^correctors .*/correctors  $CORR;/" -e "s/^source .*/source      ${SOURCE:-alpha};/" \
           $d/system/bubblesDict
    # RELTOL (optional): relTol of the solver variants (the template has 1e-9; the relTol 0 of the
    # GAMG preconditioner of PCG_GAMG is kept).
    [ -z "$RELTOL" ] || sed -i "s/^\( *relTol  *\)1e-9;/\1$RELTOL;/" $d/system/fvSolution
    [ -z "$W_RTOL" ] || sed -i "s/^\( *relTol  *\)0\.05;/\1$W_RTOL;/" $d/system/fvSolution
    [ -z "$W_ATOL" ] || sed -i "s/^\( *tolerance  *\)1e-7;/\1$W_ATOL;/" $d/system/fvSolution
    grep -q "^source  *${SOURCE:-alpha};" $d/system/bubblesDict    # template without the source entry: fail
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
# the generator (matgen), for both right-hand sides (alpha, buoyancy). Last line of
# DIR/log.check_matrix: OK / DIFFERENT (one line per source before it). Returns 0 if both are OK.
matrix_check() {
    local d=$1 app=$2 matgen=$3 src ok=0
    prep_case 24 1 $d/check24 > $d/log.check_prep 2>&1 || { echo "DIFFERENT: no check case" > $d/log.check_matrix; return 1; }
    : > $d/log.check_matrix.all
    for src in alpha buoyancy; do
        rm -f $d/check_of_$src.mtx $d/check_s2l_$src.mtx
        STEPS=4 SOURCE=$src set_dicts $d/check24 &&
            ( timeout 300 $app -case $d/check24 -solver GAMG_GS -steps 4 -correctors 1 -writeMatrix $d/check_of_$src \
                  -csv $d/check_of_$src.csv > $d/log.check_app_$src 2>&1 ) &&
            ( cd $d && timeout 300 $matgen bubbles3d 24 nb=$NBUB radius=$RADIUS rho=$RHO cfl=$CFL t=3 seed=$SEED \
                  source=$src out=$d/check_s2l_$src > $d/log.check_matgen_$src 2>&1 ) &&
            python3 $GAMG_DIR/compare_mtx.py $d/check_of_$src $d/check_s2l_$src > $d/log.check_matrix_$src 2>&1 || ok=1
        echo "$src: $(tail -n1 $d/log.check_matrix_$src 2>/dev/null || echo 'no result')" >> $d/log.check_matrix.all
    done
    # Parallel assembly (the buoyancy source has terms on the processor faces): the checksums of the
    # right-hand side of every step, printed by the application, on CHECK_NP processes against the
    # serial runs above. Skipped where decomposePar is not available (local smoke tests).
    local par=""
    if [ $ok = 0 ] && [ "${CHECK_NP:-4}" -gt 1 ] && command -v decomposePar > /dev/null; then
        sed -i "s/^numberOfSubdomains .*/numberOfSubdomains ${CHECK_NP:-4};/" $d/check24/system/decomposeParDict
        if decompose $d/check24 ${CHECK_NP:-4} hierarchical; then
            for src in alpha buoyancy; do
                STEPS=4 SOURCE=$src set_dicts $d/check24 &&
                    ( timeout 300 mpirun $MPIRUN_OPTS -n ${CHECK_NP:-4} $app -case $d/check24 -parallel -solver GAMG_GS -steps 4 \
                          -correctors 1 -csv $d/check_par_$src.csv > $d/log.check_app_par_$src 2>&1 ) &&
                    python3 - $d/log.check_app_$src $d/log.check_app_par_$src > $d/log.check_par_$src 2>&1 <<'PYEOF' || ok=1
import sys
def sums(path):
    return [tuple(float(x) for x in l.split()[-2:]) for l in open(path) if "rhs checksums" in l]
s, p = sums(sys.argv[1]), sums(sys.argv[2])
# both checksums relative to sum|b| of the serial run (sum b w may be small: b sums to ~0)
d = max(abs(a - b) / max(u[0], 1e-300) for u, v in zip(s, p) for a, b in zip(u, v)) if s and len(s) == len(p) else float("inf")
print("%s: %d steps, rhs checksums serial / parallel differ by %.1e of sum|b| (tolerance 1e-12)" % ("OK" if d <= 1e-12 else "DIFFERENT", len(s), d))
sys.exit(0 if d <= 1e-12 else 1)
PYEOF
                echo "$src, ${CHECK_NP:-4} processes: $(tail -n1 $d/log.check_par_$src 2>/dev/null || echo 'no result')" >> $d/log.check_matrix.all
            done
            par=", parallel assembly on ${CHECK_NP:-4} processes"
        else
            echo "parallel check: decomposePar failed (see $d/check24/log.decomposePar.*)" >> $d/log.check_matrix.all
            ok=1
        fi
    elif [ $ok = 0 ]; then
        echo "parallel check skipped (no decomposePar or CHECK_NP <= 1)" >> $d/log.check_matrix.all
    fi
    { cat $d/log.check_matrix.all
      if [ $ok = 0 ]; then echo "OK: both sources (alpha, buoyancy)$par"; else echo "DIFFERENT: see log.check_matrix_* / log.check_par_*"; fi
    } > $d/log.check_matrix
    return $ok
}

# src_md5: fingerprint of the sources, build files and compiler flags. Binaries built by
# Allprep_scaling.sh (cases/bin_<src_md5>) are reused by Allrun_scaling.sh only if it matches.
src_md5() {
    local repo; repo=$(cd $GAMG_DIR/../.. && pwd)
    {
        ( cd $repo && find src include test openfoam -path test/gamg -prune -o -type f \( -name '*.cpp' -o -name '*.hpp' \
              -o -name '*.h' -o -name '*.C' -o -path '*/Make/files' -o -path '*/Make/options' \) -print | sort | xargs md5sum )
        ( cd $repo && md5sum Makefile $(ls make.inc 2>/dev/null) )
        echo "$S2L_CXX $S2L_CXXFLAGS $(command -v $S2L_CXX) $WM_PROJECT_VERSION"
    } | md5sum | cut -c1-12
}

# build_s2l DIR: schwarz2lvl, matgen and the unit tests in DIR/bin (objects in DIR/build), with
# PARDISO / cluster PARDISO (MKL) and AMGCL on G100 (modules metis and intel-oneapi-mkl loaded),
# with the defaults of make.inc for SMOKE=1. Log: DIR/log.build.
build_s2l() {
    local d=$1 repo j=${MAKE_JOBS:-48}; repo=$(cd $GAMG_DIR/../.. && pwd)
    mkdir -p $d
    if [ "${SMOKE:-0}" = 1 ]; then
        ( cd $repo && make -j ${MAKE_JOBS:-4} all tests BUILD=$d/build BIN=$d/bin USE_AMGCL=1 CXX="$S2L_CXX" \
              CXXFLAGS="$S2L_CXXFLAGS" ) > $d/log.build 2>&1
        return
    fi
    libdir() { local l; for l in ${LD_LIBRARY_PATH//:/ }; do ls $l/$1.so* > /dev/null 2>&1 && { echo $l; return; }; done; }
    local metis_dir mkl_dir
    metis_dir=$(libdir libmetis)
    mkl_dir=${MKLROOT:+$MKLROOT/lib/intel64}; mkl_dir=${mkl_dir:-$(libdir libmkl_core)}
    ls $mkl_dir/libmkl_blacs_intelmpi_lp64.so* > /dev/null 2>&1 ||
        { echo "MKL BLACS for Intel MPI not found in $mkl_dir" > $d/log.build; return 1; }
    ( cd $repo && make -j $j all tests BUILD=$d/build BIN=$d/bin \
        CXX="$S2L_CXX" EIGEN_INC=external/eigen3 CXXFLAGS="$S2L_CXXFLAGS" \
        METIS_INC="$(dirname "$metis_dir")/include" METIS_LIB="-L$metis_dir -Wl,-rpath,$metis_dir -lmetis" \
        USE_MUMPS=0 USE_UMFPACK=0 USE_PARDISO=1 USE_CPARDISO=1 USE_AMGCL=1 \
        MKL_INC="$(dirname "$(dirname "$mkl_dir")")/include" \
        MKL_LIB="-L$mkl_dir -Wl,-rpath,$mkl_dir -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl" \
        MKL_BLACS_LIB="-lmkl_blacs_intelmpi_lp64" ) > $d/log.build 2>&1
}

# build_and_check DIR: build_s2l + build_app in DIR, then test_bubbles and the matrix check
# (OpenFOAM matrix = generated matrix, 24^3). Writes DIR/OK only if everything passed; returns 0 then.
build_and_check() {
    local d=$1
    rm -f $d/OK
    build_s2l $d || { echo "build of schwarz2lvl FAILED: see $d/log.build"; return 1; }
    build_app $d > $d/log.build_app 2>&1 || { echo "build of bubblyPressureFoam FAILED: see $d/log.build_app"; return 1; }
    ( cd $d && timeout 300 mpirun $MPIRUN_OPTS -n 1 $d/bin/test_bubbles > $d/log.test_bubbles 2>&1 )
    grep -q "ALL PASSED" $d/log.test_bubbles || { echo "test_bubbles FAILED: see $d/log.test_bubbles"; return 1; }
    matrix_check $d $d/bin/bubblyPressureFoam $d/bin/matgen ||
        { echo "matrix check FAILED: $(tail -n1 $d/log.check_matrix 2>/dev/null) (see $d/log.check_*)"; return 1; }
    echo "sources=$(src_md5) test_bubbles=passed matrix_check=\"$(tail -n1 $d/log.check_matrix)\" $(date '+%F %T')" > $d/OK
}
