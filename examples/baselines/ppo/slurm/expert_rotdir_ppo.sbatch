#!/bin/bash
#SBATCH --job-name=rotdir-expert
#SBATCH --partition=gpu-l40
#SBATCH --account=gpu-l40-weirdlab
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/slurm/%x-%j.out
# usage: sbatch --export=ALL,AXES=z,NAME=rotdir-expert-z expert_rotdir_ppo.sbatch   (AXES: z | xyz | z+ ...; env var ROT_AXES)
source /mmfs1/gscratch/weirdlab/will/venvs/maniskill/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export ROT_AXES=${AXES:-z} ROT_SUCCESS_ANGLE=${ANGLE:-3.14159}
cd /mmfs1/gscratch/weirdlab/will/ManiSkill/examples/baselines/ppo
python -u ppo_fast.py --env_id=RotateDirHidden-v1 --exp_name=${NAME:-rotdir-expert} --seed=1 \
  --num_envs=2048 --update_epochs=8 --num_minibatches=32 --gamma=0.99 --gae_lambda=0.95 \
  --total_timesteps=100_000_000 --num-steps=20 --num-eval-steps=200 --num_eval_envs=64 --eval_reconfiguration_freq=0 --reconfiguration_freq=0 --cudagraphs
