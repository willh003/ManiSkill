#!/bin/bash
# Generic l40 job: python -u $SCRIPT $ARGS (run from examples/baselines/ppo). usage:
#   scripts/cluster/submit.sh examples/baselines/ppo/slurm/py_a40.sbatch --job-name=<name> --export=ALL,SCRIPT=icl_transformer.py,ARGS="--kinds fcfc ..."
#SBATCH --partition=gpu-l40
#SBATCH --account=gpu-l40-weirdlab
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/slurm/%x-%j.out
source /mmfs1/gscratch/weirdlab/will/venvs/maniskill/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /mmfs1/gscratch/weirdlab/will/ManiSkill/examples/baselines/ppo
ARGS=${ARGS//@/,}  # sbatch --export splits on commas: write commas inside ARGS as @
python -u ${SCRIPT} ${ARGS}
