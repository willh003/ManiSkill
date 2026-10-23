#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --partition=ckpt
#SBATCH --account=weirdlab
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --time=00:02:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/slurm/%x-%j.out
source /mmfs1/gscratch/weirdlab/will/venvs/maniskill/bin/activate
python -c "import mani_skill; print(mani_skill.__file__)"
