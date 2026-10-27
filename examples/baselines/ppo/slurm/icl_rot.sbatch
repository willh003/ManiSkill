#!/bin/bash
# usage: sbatch --job-name=rot-<fb> --export=ALL,FB=<final_error|success|hints>,AXES=<z|xyz>[,SUFFIX=..,EXPERT=runs/..../final_ckpt.pt,EXTRA=..] slurm/icl_rot.sbatch
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
export ROT_AXES=${AXES:-xyz} ROT_SUCCESS_ANGLE=${ANGLE:-3.14159}
python -u icl_episodic.py --exp-name "rot-${FB}${SUFFIX}" --env-id RotateDirHidden-v1 --feedback "${FB}" \
  --expert-checkpoint ${EXPERT:-runs/rotdir-expert-xyz/final_ckpt.pt} --num-envs 512 --num-steps 200 --num-iterations ${ITERS:-30} \
  --d-model ${DM:-128} --n-ctx-layers ${NCL:-2} --n-dec-layers ${NDL:-2} ${EXTRA} --grad-steps ${GRAD:-2000}
