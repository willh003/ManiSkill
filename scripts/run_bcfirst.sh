#!/bin/bash
# BC attempt 1, then ICL (T1 / B) with BC rollout as context. Local, sequential.
cd /home/willhuey/research/ManiSkill/examples/baselines/ppo; P=~/venvs/maniskill/bin/python; export BCFIRST=1
$P -u icl_fair_bc.py icl-tf-dbg-c100-fcfc icl-bf-100.json
KIND=hist_c ICL_PFX=icl-tf-dbg-c100b $P -u icl_fair_bc.py icl-tf-dbg-c100-fcfc icl-bf-100-B.json
$P -u icl_fair_bc.py icl-tf-dbg-c200 icl-bf-200.json
BC_PFX=icl-tf-bcsweep BC_FILE=tf-bc-it20000.pt $P -u icl_fair_bc.py icl-tf-dbg-catabc icl-bf-500.json
$P -u icl_fair_bc.py icl-tf-dbg-catabc2k icl-bf-2k.json
echo ALLDONE
