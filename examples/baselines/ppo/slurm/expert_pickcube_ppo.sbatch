#!/bin/bash
#SBATCH --job-name=pick-expert
#SBATCH --partition=gpu-l40
#SBATCH --account=gpu-l40-weirdlab
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/slurm/%x-%j.out
source /mmfs1/gscratch/weirdlab/will/venvs/maniskill/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /mmfs1/gscratch/weirdlab/will/ManiSkill/examples/baselines/ppo
python -u ppo.py --env_id=PickCube-v1 --exp_name=pick-expert --num_envs=1024 --num_steps=50 --num_eval_steps=50 \
  --num_eval_envs=64 --gamma=0.8 --update_epochs=8 --no-capture_video --eval_freq=20 --total_timesteps=30_000_000
