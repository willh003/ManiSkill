#!/bin/bash
#SBATCH --job-name=peg-expert
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
# official ManiSkill baseline recipe for PegInsertionSide-v1 (baselines.sh)
python -u ppo_fast.py --env_id=PegInsertionSide-v1 --exp_name=peg-expert-fast --seed=1 \
  --num_envs=2048 --update_epochs=8 --num_minibatches=32 --gamma=0.97 --gae_lambda=0.95 \
  --total_timesteps=75_000_000 --num-steps=16 --num-eval-steps=100 --num_eval_envs=16 --cudagraphs
