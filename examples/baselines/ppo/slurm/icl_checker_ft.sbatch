#!/bin/bash
# Fine-tune a final wide_corner checkpoint on black (held-out) demos (icl_checker.py --finetune-ckpt). usage:
#   scripts/cluster/submit.sh examples/baselines/ppo/slurm/icl_checker_ft.sbatch --job-name=<name> --export=ALL,MODE=icl|bc,SEED=1|2|3[,SIZES=10,50,200 -> use ':' not ',': SIZES=10:50:200]
# ckpt: runs/checker-<MODE>-wide_corner-s<SEED>/checkpoints/policy_final.pt; run dir: runs/checker-ft-<MODE>-s<SEED>
#SBATCH --partition=gpu-l40
#SBATCH --account=gpu-l40-weirdlab
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/slurm/%x-%j.out
source /mmfs1/gscratch/weirdlab/will/venvs/maniskill/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd /mmfs1/gscratch/weirdlab/will/ManiSkill/examples/baselines/ppo
R=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/runs
python -u icl_checker.py --mode "${MODE}" --split wide_corner --exp-name ft-${MODE}-s${SEED} --seed ${SEED} \
  --finetune-ckpt $R/checker-${MODE}-wide_corner-s${SEED}/checkpoints/policy_final.pt --ft-sizes "$(echo ${SIZES:-10:50:200} | tr ':' ',')" \
  --expert-checkpoint $R/peg-expert-r1/strong_ckpt.pt
