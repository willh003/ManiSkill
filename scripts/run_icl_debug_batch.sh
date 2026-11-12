#!/bin/bash
# usage: run_icl_debug_batch.sh <tag> <seed>   (docs/claude/ideas/2026-11-12-icl-vs-bc-gap-debug-plan.md)
cd /home/willhuey/research/ManiSkill
P=~/venvs/maniskill/bin/python; S=examples/baselines/ppo/icl_transformer.py; tag=$1; s=$2; out=logs/runs/icl-tf-dbg-$tag-s$s
case $tag in
 perm)  $P $S --kinds fg,rg,gf,cg,gfc,cfg --ns 1000 --skip-bc --seed $s --out-dir $out ;;
 cat)   $P $S --kinds hole,goal,fcfc --cat --ns 1000 --skip-bc --seed $s --out-dir $out ;;
 long)  $P $S --kinds hole,goal --steps 30000 --ns 1000 --skip-bc --seed $s --out-dir $out ;;
 catabc) $P $S --kinds fcfc,fcfc_hist,hist_c,hist_c_noisy --cat --ns 100,1000,10000 --skip-bc --seed $s --out-dir $out ;;
 catdemos) $P $S --kinds hole,goal,fcfc --cat --bc-demos 2000 --ns 1000 --seed $s --out-dir $out ;;
 slot) $P $S --kinds hole,goal,fcfc --no-slot --ns 1000 --skip-bc --seed $s --out-dir $out ;;
 cnorm) $P $S --kinds hole,goal,fcfc --no-cnorm --ns 1000 --skip-bc --seed $s --out-dir $out ;;
 catabc2k) $P $S --kinds fcfc,fcfc_hist,hist_c,hist_c_noisy --cat --bc-demos 2000 --ns 100,1000,10000 --seed $s --out-dir $out ;;
 mlp2k) $P examples/baselines/ppo/icl_final_state.py --ctx-kind pose_sum --ctx-norm expert --ctx-drop 0 --bc-demos 2000 --ns 100,1000,10000 --steps 20000 --eval-every 5000 --seed $s --out-dir $out ;;
 demos) $P $S --kinds hole,goal --bc-demos 2000 --ns 1000 --seed $s --out-dir $out ;;
esac > logs/icl-tf-dbg-$tag-s$s.log 2>&1
