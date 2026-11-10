#!/bin/bash
# usage: run_pose_dropout.sh seed  -- plain pose ctx, expert-stat norm, BC rollouts >1m dropped, N=1000/30000
cd /home/willhuey/research/ManiSkill
~/venvs/maniskill/bin/python examples/baselines/ppo/icl_final_state.py --ctx-kind pose --ctx-norm expert --drop-outliers --ctx-drop 0 --ns 1000,30000 --steps 20000 --eval-every 5000 --seed $1 --out-dir logs/runs/icl-norm-pose-do-s$1 > logs/icl-norm-pose-do-s$1.log 2>&1
