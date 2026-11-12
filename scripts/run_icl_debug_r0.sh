#!/bin/bash
# R0/R1/R3 of docs/claude/ideas/2026-11-12-icl-vs-bc-gap-debug-plan.md: usage run_icl_debug_r0.sh seed
cd /home/willhuey/research/ManiSkill
~/venvs/maniskill/bin/python examples/baselines/ppo/icl_transformer.py --kinds hole,goal,fc --ns 1000 --skip-bc --seed $1 --out-dir logs/runs/icl-tf-dbg-r0-s$1 > logs/icl-tf-dbg-r0-s$1.log 2>&1
