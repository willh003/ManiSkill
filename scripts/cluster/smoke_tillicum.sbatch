#!/bin/bash
# Tillicum smoke test: 1/7 H200 MIG slice. Tries each Vulkan ICD (nvidia, lavapipe CPU) for env creation.
#SBATCH --job-name=smoke-til
#SBATCH --partition=gpu-h200-mig
#SBATCH --account=weirdlab
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/projects/weirdlab/will/ManiSkill/logs/slurm/%x-%j.out
source /gpfs/projects/weirdlab/will/venvs/maniskill/bin/activate
nvidia-smi -L
cat > /tmp/smoke_$$.py <<'PY'
import sys, torch, gymnasium as gym, mani_skill.envs
rb = sys.argv[1]
env = gym.make("PegInsertionSide-v1", num_envs=64, obs_mode="state", sim_backend="gpu", render_backend=rb)
env.reset(seed=0)
for _ in range(20): env.step(env.action_space.sample())
print("OK state step, render_backend", rb)
PY
for icd in none nvidia lvp; do
  echo "=== ICD=$icd"
  if [ $icd = none ]; then unset VK_ICD_FILENAMES; else export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/${icd}_icd.x86_64.json; fi
  for rb in none gpu; do echo "--- render_backend=$rb"; python /tmp/smoke_$$.py $rb 2>&1 | grep -vE "^\s+File|^\s+\^" | tail -6; done
done
