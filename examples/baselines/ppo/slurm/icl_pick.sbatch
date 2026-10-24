#!/bin/bash
# usage: sbatch --job-name=pick-<fb> --export=ALL,FB=<final_error|success|hints>[,SUFFIX=-x,ITERS=n,GRAD=n] slurm/icl_pick.sbatch
#SBATCH --partition=gpu-l40
#SBATCH --account=gpu-l40-weirdlab
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/slurm/%x-%j.out
source /mmfs1/gscratch/weirdlab/will/venvs/maniskill/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PICK_CUBE_OBS_BIAS=0.05
cd /mmfs1/gscratch/weirdlab/will/ManiSkill/examples/baselines/ppo
python -u icl_episodic.py --exp-name "pick-${FB}${SUFFIX}" --env-id PickCubeBias-v1 --feedback "${FB}" \
  --expert-checkpoint runs/pick-expert/final_ckpt.pt --num-envs 512 --num-steps 50 --num-iterations ${ITERS:-30} --grad-steps ${GRAD:-2000}
