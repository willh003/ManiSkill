#!/bin/bash
# Episodic ICL student (icl_episodic.py). usage:
#   sbatch --job-name=<name> --export=ALL,ENV_ID=<env>,NAME=<run name>,EXPERT=runs/<expert>/final_ckpt.pt,STEPS=<episode length>[,FB=final_error|success|hints,ITERS=30,GRAD=2000,EXTRA="--hide-hole",<env knobs>] slurm/icl.sbatch
# Run dir: runs/icl-<NAME>. Env knobs read by the task envs (ROT_AXES, PICK_CUBE_OBS_BIAS, PUSHT_GOAL_POS_RANGE, ...) are passed through --export, e.g.
#   PushTGoal-v1: ENV_ID=PushTGoal-v1,STEPS=100,PUSHT_GOAL_POS_RANGE=0.03,PUSHT_GOAL_YAW_DEG=45     PickCubeBias-v1: STEPS=50,PICK_CUBE_OBS_BIAS=0.05
#   PegInsertionSideBias-v1: STEPS=100,EXTRA=--hide-hole                                             RotateDirHidden-v1: STEPS=200,ROT_AXES=xyz
# Note: sbatch --export splits on commas, so list-valued args (--ctx-policies) must use ':' as the separator.
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
python -u icl_episodic.py --exp-name "${NAME}" --env-id "${ENV_ID}" --feedback "${FB:-final_error}" --expert-checkpoint "${EXPERT}" \
  --num-envs 512 --num-steps "${STEPS}" --num-iterations ${ITERS:-30} --grad-steps ${GRAD:-2000} \
  --d-model ${DM:-128} --n-ctx-layers ${NCL:-2} --n-dec-layers ${NDL:-2} ${EXTRA}
