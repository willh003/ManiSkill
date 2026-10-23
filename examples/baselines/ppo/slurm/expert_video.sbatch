#!/bin/bash
#SBATCH --job-name=expert-video
#SBATCH --partition=gpu-a40
#SBATCH --account=gpu-a40-weirdlab
#SBATCH --gres=gpu:a40:1
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/%x-%j.out
source /mmfs1/gscratch/weirdlab/will/venvs/maniskill/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PUSHT_GOAL_POS_RANGE=0.03 PUSHT_GOAL_YAW_DEG=45
cd /mmfs1/gscratch/weirdlab/will/ManiSkill/examples/baselines/ppo
mkdir -p renders
python -u expert_video.py runs/pr-full/final_ckpt.pt renders/goal_expert.mp4
