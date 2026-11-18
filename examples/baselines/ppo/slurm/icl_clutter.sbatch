#!/bin/bash
#SBATCH --job-name=icl-clutter
#SBATCH --partition=gpu-l40
#SBATCH --account=gpu-l40-weirdlab
#SBATCH --gres=gpu:l40:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/slurm/%x-%j.out
# icl_clutter.py: object-centric diffusion BC vs ICL (+ on-policy rounds) on PickClutterYCBCond-v1.
# Env: DEMOS (layouts), SEED, ROUNDS, NAME (run dir name), EXTRA (more args).
# e.g. sbatch --export=ALL,DEMOS=100,SEED=1,ROUNDS=4 slurm/icl_clutter.sbatch
# Submitted to several partitions under one job name (scripts/cluster/submit_clutter.sh): the first copy to start takes a lock
# and cancels its pending siblings.
ROOT=/mmfs1/gscratch/weirdlab/will/ManiSkill
LOCK=$ROOT/logs/onpolicy-locks/$SLURM_JOB_NAME
mkdir -p $ROOT/logs/onpolicy-locks
if ! mkdir $LOCK 2>/dev/null; then
  sleep 10
  other=$(cat $LOCK/jobid 2>/dev/null)
  if [ -n "$other" ] && [ "$other" != "$SLURM_JOB_ID" ] && squeue -h -j "$other" -t RUNNING 2>/dev/null | grep -q .; then
    echo "sibling job $other already runs $SLURM_JOB_NAME; exiting"; exit 0
  fi
fi
echo $SLURM_JOB_ID > $LOCK/jobid
squeue -h -u $USER -n $SLURM_JOB_NAME -t PENDING -o %i | grep -v "^$SLURM_JOB_ID$" | xargs -r scancel
echo "job $SLURM_JOB_ID on $(hostname) partition $SLURM_JOB_PARTITION: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
source /mmfs1/gscratch/weirdlab/will/venvs/maniskill/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
DEMOS=${DEMOS:-200}; SEED=${SEED:-1}; ROUNDS=${ROUNDS:-4}
NAME=${NAME:-icl-clutter-diff-d${DEMOS}-r${ROUNDS}-s${SEED}}
cd /mmfs1/gscratch/weirdlab/will/ManiSkill/examples/baselines/ppo
python -u icl_clutter.py --expert-checkpoint ../../../logs/runs/clutter-expert-s1/final_ckpt.pt \
  --demos $DEMOS --seed $SEED --rounds $ROUNDS --num-envs 256 \
  --arch obj --head diff --chunk 8 --exec-steps 4 --steps 20000 \
  --out-dir ../../../logs/runs/$NAME $EXTRA
