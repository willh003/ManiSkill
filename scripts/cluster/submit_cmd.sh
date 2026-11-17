#!/bin/bash
# Push, then run one command on a klone GPU (weirdlab l40 + ckpt a40/l40/l40s; first to start cancels the rest).
# Usage: submit_cmd.sh <job name> "<command, run from examples/baselines/ppo; \$ROOT = repo on klone>"   TIME=hh:mm:ss to override 4h.
# Output: logs/slurm/<job name>.log on klone.
source "$(dirname "$0")/config.sh"
"$(dirname "$0")/push.sh" >/dev/null || exit $?
SB=examples/baselines/ppo/slurm/cmd.sbatch
n=$1; b=$(printf '%s' "$2" | base64 -w0); ex="--export=ALL,CMD_B64=$b"
[ -n "$TIME" ] && ex+=" --time=$TIME"
cmd="cd $REMOTE_DIR && mkdir -p logs/slurm; sbatch --parsable -J $n -A gpu-l40-weirdlab -p gpu-l40 --gres=gpu:l40:1 $ex $SB"
for g in a40 l40 l40s; do cmd+="; sbatch --parsable -J $n -A ckpt-weirdlab -p ckpt-all --gres=gpu:$g:1 $ex $SB"; done
ssh "$REMOTE_HOST" "$cmd"
