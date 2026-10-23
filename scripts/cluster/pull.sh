#!/bin/bash
# Pull results from klone into local logs/. Usage: pull.sh [--no-ckpt] [run-name ...]
# With no run names, pulls all of logs/{runs,renders,slurm}.
source "$(dirname "$0")/config.sh"
check_ssh
args=(-az --itemize-changes); runs=()
for a in "$@"; do
  case $a in
    --no-ckpt) args+=(--exclude='*.pt' --exclude='*.pth') ;;
    *) runs+=("$a") ;;
  esac
done
mkdir -p "$LOCAL_DIR/logs"
if [ ${#runs[@]} -eq 0 ]; then
  rsync "${args[@]}" "$REMOTE_HOST:$REMOTE_DIR/logs/" "$LOCAL_DIR/logs/"
else
  mkdir -p "$LOCAL_DIR/logs/runs" "$LOCAL_DIR/logs/slurm"
  for r in "${runs[@]}"; do rsync "${args[@]}" "$REMOTE_HOST:$REMOTE_DIR/logs/runs/$r" "$LOCAL_DIR/logs/runs/"; done
  rsync "${args[@]}" "$REMOTE_HOST:$REMOTE_DIR/logs/slurm/" "$LOCAL_DIR/logs/slurm/"
fi
