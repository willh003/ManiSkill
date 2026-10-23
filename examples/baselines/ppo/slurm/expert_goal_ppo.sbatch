#!/bin/bash
#SBATCH --job-name=expert-goal-ppo
#SBATCH --partition=gpu-l40
#SBATCH --account=gpu-l40-weirdlab
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/slurm/%x-%j.out
source /mmfs1/gscratch/weirdlab/will/venvs/maniskill/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PUSHT_GOAL_POS_RANGE=0.03 PUSHT_GOAL_YAW_DEG=45
cd /mmfs1/gscratch/weirdlab/will/ManiSkill/examples/baselines/ppo
python -u ppo.py --env_id=PushTGoal-v1 --exp_name=pr-full --num_envs=1024 --num_steps=100 --num_eval_steps=100 \
  --num_eval_envs=64 --gamma=0.99 --update_epochs=8 --no-capture_video --eval_freq=20 --total_timesteps=75_000_000
