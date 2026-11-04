#!/bin/bash
# Checkerboard hold-out test (icl_checker.py). usage:
#   scripts/cluster/submit.sh examples/baselines/ppo/slurm/icl_checker.sbatch --job-name=<name> --export=ALL,MODE=icl|bc,SEED=1,NAME=<run name>[,SPLIT=checker|wide_ring|wide_half|wide_corner,ITERS=30,GRAD=2000]
# Run dir: runs/checker-<NAME>. Expert: runs/peg-expert-r1/strong_ckpt.pt (strong peg expert).
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
python -u icl_checker.py --mode "${MODE}" --exp-name "${NAME}" --seed ${SEED:-1} --split ${SPLIT:-checker} --num-iterations ${ITERS:-30} --grad-steps ${GRAD:-2000} \
  --expert-checkpoint /mmfs1/gscratch/weirdlab/will/ManiSkill/logs/runs/peg-expert-r1/strong_ckpt.pt
