#!/bin/bash
# Run one arbitrary command on a GPU (scripts/cluster/submit_cmd.sh). Same first-to-start-wins prologue as icl_onpolicy.sbatch.
# Env: CMD_B64 = base64 of the command, run from examples/baselines/ppo; output to logs/slurm/<job name>.log
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/weirdlab/will/ManiSkill/logs/slurm/%x-%j.out
export ROOT=/mmfs1/gscratch/weirdlab/will/ManiSkill
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
CMD=$(echo "$CMD_B64" | base64 -d)
echo "+ $CMD"
bash -c "$CMD" > $ROOT/logs/slurm/$NAME.log 2>&1
echo "exit $?"
