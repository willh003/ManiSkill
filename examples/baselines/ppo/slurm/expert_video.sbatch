#!/bin/bash
#SBATCH --job-name=expert-video
#SBATCH --partition=gpu-l40
#SBATCH --account=gpu-l40-weirdlab
#SBATCH --gres=gpu:l40:1
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/slurm/%x-%j.out
source /mmfs1/gscratch/weirdlab/will/venvs/maniskill/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PUSHT_GOAL_POS_RANGE=0.03 PUSHT_GOAL_YAW_DEG=45
cd /mmfs1/gscratch/weirdlab/will/ManiSkill/examples/baselines/ppo
python -u expert_video.py runs/pr-full/final_ckpt.pt ../../../logs/renders/pusht-goal/goal_expert.mp4
