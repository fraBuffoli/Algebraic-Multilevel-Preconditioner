#!/bin/bash
# Submits the scaling campaign from test/gamg:
#   Allprep_scaling.sh (1 node: meshes, binaries, checks), then Allrun_scaling.sh on 5, 7 and 9
#   nodes (240, 336, 432 processes), each starting only if the preparation succeeded.
#   NODES="5 7 9" by default; SERIAL=1 runs the three jobs one after the other (no two jobs of the
#   campaign at the same time on the file system and the network; longer in the queue).
#   PREP=0 skips the preparation (cases and binaries already there).
# Then: python3 report_scaling.py run_scaling_np*_<jobid> ...  (see README.md)
cd "$(dirname "$0")" || exit 1
NODES=${NODES:-"5 7 9"}
NPS=""; for n in $NODES; do NPS="$NPS $((n * 48))"; done
dep=""
if [ "${PREP:-1}" = 1 ]; then
    jid=$(NPS="${NPS# }" sbatch --parsable --export=ALL Allprep_scaling.sh) || exit 1
    echo "Allprep_scaling.sh: job $jid (process counts:$NPS)"
    dep="--dependency=afterok:$jid"
fi
for n in $NODES; do
    np=$((n * 48))
    j=$(sbatch --parsable $dep --nodes=$n --ntasks-per-node=48 --job-name=s2l_scal_$np \
        --output=allrun_scaling_$np.out --error=allrun_scaling_$np.err Allrun_scaling.sh) || exit 1
    echo "Allrun_scaling.sh NP=$np: job $j"
    if [ "${SERIAL:-0}" = 1 ]; then
        if [ -n "$jid" ]; then dep="--dependency=afterok:$jid,afterany:$j"; else dep="--dependency=afterany:$j"; fi
    fi
done
echo "If the preparation fails, the dependent jobs stay pending: scancel them."
