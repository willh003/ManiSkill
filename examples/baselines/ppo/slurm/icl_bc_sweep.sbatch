#!/bin/bash
# Transformer BC (visible goal, 500 expert demos): checkpoint + val loss + success every 500 steps (icl_transformer.py --bc-only). usage:
#   scripts/cluster/submit.sh examples/baselines/ppo/slurm/icl_bc_sweep.sbatch --job-name=tf-bcsweep-s<seed> --export=ALL,SEED=1
# Run dir: logs/runs/icl-tf-bcsweep-s<SEED> (checkpoints/tf-bc-it*.pt, trace_bc.json = (step, success, val))
#SBATCH --partition=gpu-a40
#SBATCH --account=gpu-a40-weirdlab
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/slurm/%x-%j.out
source /mmfs1/gscratch/weirdlab/will/venvs/maniskill/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /mmfs1/gscratch/weirdlab/will/ManiSkill/examples/baselines/ppo
python -u icl_transformer.py --bc-only --kinds fcfc --steps ${STEPS:-20000} --eval-every ${EVERY:-500} --seed ${SEED:-1} \
  --out-dir /mmfs1/gscratch/weirdlab/will/ManiSkill/logs/runs/icl-tf-bcsweep-s${SEED:-1}
