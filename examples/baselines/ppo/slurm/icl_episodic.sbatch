#!/bin/bash
#SBATCH --job-name=icl-terminate
#SBATCH --partition=gpu-l40s
#SBATCH --account=gpu-l40s-weirdlab
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/%x-%j.out
source /mmfs1/gscratch/weirdlab/will/venvs/maniskill/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PUSHT_GOAL_POS_RANGE=0.03 PUSHT_GOAL_YAW_DEG=45
cd /mmfs1/gscratch/weirdlab/will/ManiSkill/examples/baselines/ppo
python -u icl_episodic.py --exp-name terminate --expert-checkpoint runs/pr-full/final_ckpt.pt --num-envs 512 --num-iterations 20
