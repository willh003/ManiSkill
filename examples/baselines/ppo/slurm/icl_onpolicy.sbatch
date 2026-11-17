#!/bin/bash
# On-policy ICL loop (icl_onpolicy.py), 3 seeds in parallel on one GPU. Submitted by scripts/cluster/submit_onpolicy.sh to several
# partitions under one job name: the first copy to start takes a lock and cancels its pending siblings. Resumable (state.pt), so
# ckpt preemption + requeue is safe. Env: KIND PAIR SUP DEMOS [SEEDS=1@2@3] [ROUNDS=6]. Run dirs: logs/runs/icl-op-<KIND>-<PAIR>-<SUP>-d<DEMOS>-s<seed>
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/slurm/%x-%j.out
ROOT=/mmfs1/gscratch/weirdlab/will/ManiSkill
NAME=$SLURM_JOB_NAME
LOCK=$ROOT/logs/onpolicy-locks/$NAME
mkdir -p $ROOT/logs/onpolicy-locks
if ! mkdir $LOCK 2>/dev/null; then
  sleep 10
  other=$(cat $LOCK/jobid 2>/dev/null)
  if [ -n "$other" ] && [ "$other" != "$SLURM_JOB_ID" ] && squeue -h -j "$other" -t RUNNING 2>/dev/null | grep -q .; then
    echo "sibling job $other already runs $NAME; exiting"; exit 0
  fi
fi
echo $SLURM_JOB_ID > $LOCK/jobid
squeue -h -u $USER -n $NAME -t PENDING -o %i | grep -v "^$SLURM_JOB_ID$" | xargs -r scancel
echo "job $SLURM_JOB_ID on $(hostname) partition $SLURM_JOB_PARTITION: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
source /mmfs1/gscratch/weirdlab/will/venvs/maniskill/bin/activate
export OMP_NUM_THREADS=2
cd $ROOT/examples/baselines/ppo
for s in ${SEEDS//@/ }; do
  python -u icl_onpolicy.py --kind $KIND --pairing $PAIR --supervise $SUP --demos $DEMOS --rounds ${ROUNDS:-6} --seed $s \
    --out-dir $ROOT/logs/runs/icl-op-$KIND-$PAIR-$SUP-d$DEMOS-s$s > $ROOT/logs/slurm/$NAME-s$s.log 2>&1 &
done
wait
grep -c ALLDONE $ROOT/logs/slurm/$NAME-s*.log
