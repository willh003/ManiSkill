#!/bin/bash
# Push, then submit on-policy ICL jobs. Each config goes to weirdlab l40 and to ckpt a40 / l40 / l40s under one job name
# (the copy that starts first cancels the others, see slurm/icl_onpolicy.sbatch).
# Env: GOALREF=hole (c = hole pose - f; run and job names get -hole), SEEDS (1@2@3 = 3 seeds in one job; a single seed = one job per seed), ROUNDS, TIME (sbatch limit; short limits schedule sooner).
# Usage: submit_onpolicy.sh "<kind> <pairing> <supervise> <demos> [fresh]" ...  (fresh: roll out on new tasks each round)     e.g. submit_onpolicy.sh "fc matched demo 100" "goal matched demo 2000"
source "$(dirname "$0")/config.sh"
"$(dirname "$0")/push.sh" >/dev/null || exit $?
SB=examples/baselines/ppo/slurm/icl_onpolicy.sbatch
cmd="cd $REMOTE_DIR && mkdir -p logs/slurm logs/runs"
for cfg in "$@"; do
  set -- $cfg; h=$([ "${GOALREF:-peg}" = hole ] && echo -hole); n=op-$1-$2-$3${5:+-$5}$h-d$4; ex="--export=ALL,KIND=$1,PAIR=$2,SUP=$3,DEMOS=$4,SEEDS=${SEEDS:-1@2@3},ROUNDS=${ROUNDS:-6},ROLLOUT=${5:-demo},GOALREF=${GOALREF:-peg}"
  [[ "${SEEDS:-1@2@3}" != *@* ]] && n+=-s$SEEDS  # one seed per job: the job name (and lock) must be per seed
  [ -n "$TIME" ] && ex+=" --time=$TIME"
  cmd+="; sbatch --parsable -J $n -A gpu-l40-weirdlab -p gpu-l40 --gres=gpu:l40:1 $ex $SB"
  for g in a40 l40 l40s; do cmd+="; sbatch --parsable -J $n -A ckpt-weirdlab -p ckpt-all --gres=gpu:$g:1 $ex $SB"; done
done
ssh "$REMOTE_HOST" "$cmd"
