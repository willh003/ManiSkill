#!/bin/bash
# Push, then submit BC-first re-evals of on-policy runs: one job per config (its 3 seeds in parallel on one GPU), each sent to
# weirdlab l40 and ckpt a40 / l40 / l40s (first to start cancels the rest).
# Usage: submit_onpolicy_eval.sh <run prefix | run name> ...   e.g. icl-op-fc-matched-demo-d100 (seeds 1-3 in one job) or
#        icl-op-fc-matched-demo-d100-s2 (one run per job). TIME=hh:mm:ss overrides the sbatch time limit (short limits schedule sooner).
source "$(dirname "$0")/config.sh"
"$(dirname "$0")/push.sh" >/dev/null || exit $?
SB=examples/baselines/ppo/slurm/icl_onpolicy_eval.sbatch
cmd="cd $REMOTE_DIR && mkdir -p logs/slurm"
for p in "$@"; do
  n=ev-${p#icl-op-}
  if [[ $p =~ -s[0-9]+$ ]]; then ex="--export=ALL,RUNS=$p"; else ex="--export=ALL,RUNS=$p-s1@$p-s2@$p-s3"; fi
  [ -n "$TIME" ] && ex+=" --time=$TIME"
  cmd+="; sbatch --parsable -J $n -A gpu-l40-weirdlab -p gpu-l40 --gres=gpu:l40:1 $ex $SB"
  for g in a40 l40 l40s; do cmd+="; sbatch --parsable -J $n -A ckpt-weirdlab -p ckpt-all --gres=gpu:$g:1 $ex $SB"; done
done
ssh "$REMOTE_HOST" "$cmd"
