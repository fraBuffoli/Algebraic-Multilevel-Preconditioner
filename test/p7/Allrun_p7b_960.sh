#!/bin/bash
#SBATCH --nodes=20                   	# 20 nodes
#SBATCH --ntasks-per-node=48         	# 48 tasks per node -> 960 MPI processes
#SBATCH --exclusive                  	# whole nodes: clean timings
#SBATCH --partition=g100_usr_prod    	# server type
#SBATCH --error=allrun_p7b_960.err    	# standard error file
#SBATCH --output=allrun_p7b_960.out   	# standard output file
#SBATCH --account=IscrC_EXTA-PMF       	# account name
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=francesco.buffoli@mail.polimi.it
#SBATCH --job-name=s2l_p7b960
#SBATCH --time=00:30:00
# -------------------------------------------------------------------------------- #
# P7b at 960 processes: same problems as Allrun_p7b.sh with 4x more subdomains
#   bub192: ~7k cells per process   (exact solves already cheap: does AMG still help?)
#   bub256: ~17k cells per process  (where the first campaign had 2 AMG levels and a slow setup:
#                                    checks the fix --amg-coarse-enough 500)
# No unit tests (they run in the 240-process job), fewer variants. Runs Allrun_p7b.sh with:
#   192^3: ref, sa     256^3: ref, sp, sa
#
#   1. from test/p7:               sbatch Allrun_p7b_960.sh             (~12 min, ~200 core-hours)
#   2. summary + plots:            python3 report_p7.py run_p7b_<jobid>
# -------------------------------------------------------------------------------- #
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
export NP=${NP:-960}
export RUN_CHECK=${RUN_CHECK:-0}
export VARIANTS_A=${VARIANTS_A:-"ref sa"}
export VARIANTS_B=${VARIANTS_B:-"ref sp sa"}
exec bash ./Allrun_p7b.sh
