#!/bin/bash
# BC-first re-eval of icl_onpolicy.py runs (icl_onpolicy_bcfirst.py). Submitted by scripts/cluster/submit_onpolicy_eval.sh to several
# partitions under one job name; the first copy to start cancels its pending siblings. Env: RUNS = run names separated by @, each
# evaluated by its own process on the one GPU. Output: logs/runs/<run>/results_bcfirst.json
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=6:00:00
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
export OMP_NUM_THREADS=1
cd $ROOT/examples/baselines/ppo
for r in ${RUNS//@/ }; do
  python -u icl_onpolicy_bcfirst.py $ROOT/logs/runs/$r > $ROOT/logs/slurm/$NAME-$r.log 2>&1 &
done
wait
grep -c ALLDONE $ROOT/logs/slurm/$NAME-*.log
