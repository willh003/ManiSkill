#!/bin/bash
#SBATCH --job-name=peg-expert-resume
#SBATCH --partition=gpu-l40
#SBATCH --account=gpu-l40-weirdlab
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=16:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/slurm/%x-%j.out
# Continue the peg expert from runs/peg-expert-fast/final_ckpt.pt. Usage: sbatch --export=ALL,NAME=<run>,LR=<lr>,SEED=<seed> [--job-name=..] this.sbatch
source /mmfs1/gscratch/weirdlab/will/venvs/maniskill/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /mmfs1/gscratch/weirdlab/will/ManiSkill/examples/baselines/ppo
python -u ppo_fast.py --env_id=PegInsertionSide-v1 --exp_name=${NAME} --seed=${SEED:-2} \
  --checkpoint=runs/peg-expert-fast/final_ckpt.pt --learning_rate=${LR} --anneal_lr \
  --num_envs=2048 --update_epochs=8 --num_minibatches=32 --gamma=0.97 --gae_lambda=0.95 \
  --total_timesteps=150_000_000 --num-steps=16 --num-eval-steps=100 --num_eval_envs=64 --cudagraphs --no-capture_video
