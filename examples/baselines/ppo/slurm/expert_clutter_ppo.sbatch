#!/bin/bash
#SBATCH --job-name=clutter-expert
#SBATCH --partition=gpu-l40
#SBATCH --account=gpu-l40-weirdlab
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/slurm/%x-%j.out
# Target-conditioned PPO expert for PickClutterYCBCond-v1 (PickSingleYCB recipe from baselines.sh, 100M steps).
# Env: NAME (run name, default clutter-expert-s$SEED), SEED (default 1), STEPS (default 100_000_000)
source /mmfs1/gscratch/weirdlab/will/venvs/maniskill/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
SEED=${SEED:-1}
NAME=${NAME:-clutter-expert-s$SEED}
cd /mmfs1/gscratch/weirdlab/will/ManiSkill/examples/baselines/ppo
python -u ppo_fast.py --env_id=PickClutterYCBCond-v1 --exp_name=$NAME --seed=$SEED \
  --num_envs=4096 --num-steps=16 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=${STEPS:-100_000_000} --num-eval-steps=75 --num_eval_envs=64 --eval_freq=25 \
  --no-capture-video --cudagraphs
