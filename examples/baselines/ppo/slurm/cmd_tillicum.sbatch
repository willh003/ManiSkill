#!/bin/bash
# Run one arbitrary command on a Tillicum H200 MIG slice (scripts/cluster/submit_cmd.sh with CLUSTER=tillicum).
# State-only: MIG has no Vulkan, so ManiSkill runs with lavapipe ICD + render_backend=none. Max 2 CPUs per GPU.
# Env: CMD_B64 = base64 of the command, run from examples/baselines/ppo; output to logs/slurm/<job name>.log
#SBATCH --partition=gpu-h200-mig
#SBATCH --account=weirdlab
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=30G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=/gpfs/projects/weirdlab/will/ManiSkill/logs/slurm/%x-%j.out
export ROOT=/gpfs/projects/weirdlab/will/ManiSkill
NAME=$SLURM_JOB_NAME
echo "job $SLURM_JOB_ID on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader)"
source /gpfs/projects/weirdlab/will/venvs/maniskill/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
export MANISKILL_RENDER_BACKEND=none
export OMP_NUM_THREADS=2
cd $ROOT/examples/baselines/ppo
CMD=$(echo "$CMD_B64" | base64 -d)
echo "+ $CMD"
bash -c "$CMD" > $ROOT/logs/slurm/$NAME.log 2>&1
echo "exit $?"
