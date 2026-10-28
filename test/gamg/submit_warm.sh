#!/bin/bash
# Submits the warm-start campaign from test/gamg:
#   Allprep_scaling.sh (1 node: meshes and decompositions of SIZES on the process counts of NODES,
#   binaries, matrix check of both right-hand sides), then Allrun_warm.sh on 5 and 9 nodes (240 and
#   432 processes). By default the two runs go one after the other (SERIAL=1: no two jobs of the
#   campaign at the same time on the file system and the network); SERIAL=0 submits them together.
#   PREP=0 skips the preparation (Allrun_warm.sh then builds the binaries and prepares missing cases
#   itself). Any variable of Allrun_warm.sh (SIZES, STEPS, SOURCE, WARM, RTOL, ATOL, ...) is passed on.
# Then: python3 report_warm.py run_warm_np*_<jobid> ...  (see README.md)
cd "$(dirname "$0")" || exit 1
NODES=${NODES:-"5 9"}
export SIZES=${SIZES:-"192 256 320"}
NPS=""; for n in $NODES; do NPS="$NPS $((n * 48))"; done
dep="" jid=""
if [ "${PREP:-1}" = 1 ]; then
    jid=$(NPS="${NPS# }" sbatch --parsable --export=ALL Allprep_scaling.sh) || exit 1
    echo "Allprep_scaling.sh: job $jid (sizes $SIZES, process counts:$NPS)"
    dep="--dependency=afterok:$jid"
fi
for n in $NODES; do
    np=$((n * 48))
    j=$(sbatch --parsable --export=ALL $dep --nodes=$n --ntasks-per-node=48 --job-name=s2l_warm_$np \
        --output=allrun_warm_$np.out --error=allrun_warm_$np.err Allrun_warm.sh) || exit 1
    echo "Allrun_warm.sh NP=$np: job $j"
    if [ "${SERIAL:-1}" = 1 ]; then
        if [ -n "$jid" ]; then dep="--dependency=afterok:$jid,afterany:$j"; else dep="--dependency=afterany:$j"; fi
    fi
done
echo "If the preparation fails, the dependent jobs stay pending: scancel them."
