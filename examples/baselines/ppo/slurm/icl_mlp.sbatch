#!/bin/bash
# MLP pose_sum reference at N=10000 (icl_final_state.py), to compare with the transformer grid. usage:
#   scripts/cluster/submit.sh examples/baselines/ppo/slurm/icl_mlp.sbatch --job-name=mlp-ps-s<seed> --export=ALL,SEED=1
#SBATCH --partition=gpu-l40
#SBATCH --account=gpu-l40-weirdlab
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/slurm/%x-%j.out
source /mmfs1/gscratch/weirdlab/will/venvs/maniskill/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /mmfs1/gscratch/weirdlab/will/ManiSkill/examples/baselines/ppo
python -u icl_final_state.py --ctx-kind pose_sum --ctx-norm expert --ctx-drop 0 --ns 10000 --steps 20000 --eval-every 5000 --seed ${SEED:-1} \
  --out-dir /mmfs1/gscratch/weirdlab/will/ManiSkill/logs/runs/icl-norm-pose_sum-N10000-s${SEED:-1}
