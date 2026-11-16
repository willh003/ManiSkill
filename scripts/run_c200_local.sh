#!/bin/bash
# usage: run_c200_local.sh <cb|mlp> <seed>  -- 200-demo biased-C / MLP runs (local)
cd /home/willhuey/research/ManiSkill; P=~/venvs/maniskill/bin/python; s=$2
case $1 in
 cb)  $P examples/baselines/ppo/icl_transformer.py --kinds hist_c_noisy --cat --bc-demos 200 --skip-bc --noise-lo 1.1 --noise-hi 1.3 --ns 100,1000,10000 --seed $s --out-dir logs/runs/icl-tf-dbg-c200cb-s$s ;;
 mlp) $P examples/baselines/ppo/icl_final_state.py --ctx-kind pose_sum --ctx-norm expert --ctx-drop 0 --bc-demos 200 --ns 100,1000,10000 --steps 20000 --eval-every 5000 --seed $s --out-dir logs/runs/icl-tf-dbg-mlp200-s$s ;;
esac
