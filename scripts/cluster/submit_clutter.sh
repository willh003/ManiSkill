#!/bin/bash
# Push, then submit icl_clutter jobs to weirdlab l40 and ckpt a40/l40/l40s under one job name per config (first to start
# cancels the rest; see slurm/icl_clutter.sbatch).
# Usage: submit_clutter.sh "<demos> <seed> <rounds> [name-suffix] [extra args...]" ...   Env: TIME (sbatch limit)
source "$(dirname "$0")/config.sh"
"$(dirname "$0")/push.sh" >/dev/null || exit $?
SB=examples/baselines/ppo/slurm/icl_clutter.sbatch
cmd="cd $REMOTE_DIR && mkdir -p logs/slurm logs/runs"
for cfg in "$@"; do
  set -- $cfg; d=$1; s=$2; r=$3; suf=${4:-}; shift 4 2>/dev/null || shift $#
  n=icl-clutter-diff${suf:+-$suf}-d$d-r$r-s$s
  ex="--export=ALL,DEMOS=$d,SEED=$s,ROUNDS=$r,NAME=$n,EXTRA='$*'"
  [ -n "$TIME" ] && ex+=" --time=$TIME"
  cmd+="; sbatch --parsable -J $n -A gpu-l40-weirdlab -p gpu-l40 --gres=gpu:l40:1 $ex $SB"
  for g in a40 l40 l40s; do cmd+="; sbatch --parsable -J $n -A ckpt-weirdlab -p ckpt-all --gres=gpu:$g:1 $ex $SB"; done
done
ssh "$REMOTE_HOST" "$cmd"
