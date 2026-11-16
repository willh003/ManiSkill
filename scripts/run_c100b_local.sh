#!/bin/bash
# usage: run_c100b_local.sh <seed>  -- retrain variant B (hist_c) at 100 demos, N=1000 only, for the fair-BC eval (klone checkpoints unavailable)
cd /home/willhuey/research/ManiSkill
~/venvs/maniskill/bin/python examples/baselines/ppo/icl_transformer.py --kinds hist_c --cat --bc-demos 100 --skip-bc --ns 1000 --seed $1 --out-dir logs/runs/icl-tf-dbg-c100b-s$1
