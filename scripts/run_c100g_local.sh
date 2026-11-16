#!/bin/bash
# usage: run_c100g_local.sh <seed> -- goal-only context [g=f+c], 100 demos, concat pathway, N=1000; then BC-first eval
cd /home/willhuey/research/ManiSkill
~/venvs/maniskill/bin/python examples/baselines/ppo/icl_transformer.py --kinds goal --cat --bc-demos 100 --skip-bc --ns 1000 --seed $1 --out-dir logs/runs/icl-tf-dbg-c100g-s$1
