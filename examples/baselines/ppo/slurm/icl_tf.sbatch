#!/bin/bash
# Transformer offline-ICL ablation (icl_transformer.py). usage:
#   scripts/cluster/submit.sh examples/baselines/ppo/slurm/icl_tf.sbatch --job-name=tf-<kind>-s<seed> --export=ALL,KINDS=fcfc,SEED=1[,NS=100,1000,10000][,SKIP_BC=--skip-bc]
# Run dir: logs/runs/icl-tf-<KINDS>-s<SEED>
#SBATCH --partition=gpu-l40
#SBATCH --account=gpu-l40-weirdlab
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/slurm/%x-%j.out
source /mmfs1/gscratch/weirdlab/will/venvs/maniskill/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /mmfs1/gscratch/weirdlab/will/ManiSkill/examples/baselines/ppo
python -u icl_transformer.py --kinds "${KINDS}" --seed ${SEED:-1} --ns "${NS:-100,1000,10000}" ${SKIP_BC:-} \
  --out-dir /mmfs1/gscratch/weirdlab/will/ManiSkill/logs/runs/icl-tf-${KINDS}-s${SEED:-1}
