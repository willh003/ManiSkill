# On-policy outer loop for hidden-hole peg ICL (2026-11-16)
Script `examples/baselines/ppo/icl_onpolicy.py`, plots `icl_onpolicy_plot.py` -> `logs/renders/icl-onpolicy/`, job `slurm/icl_onpolicy.sbatch`, submit `scripts/cluster/submit_onpolicy.sh`. Runs `logs/runs/icl-op-<kind>-<pairing>-<supervise>-d<demos>-s<seed>/` (results.json, curves.json, state.pt, checkpoints `tf-bc.pt`, `round<r>.pt`). Tracker experiment `icl-onpolicy-loop`.

## Why
The 2026-11-11..15 runs showed the ICL policy ignores the failure feedback: it was trained once on failures of a visible-hole BC, each paired with a random other task's expert demo. This adds an outer loop of on-policy collection.

## Recipe
- E expert demos (100 / 200 / 2000) + 200 held-out val demos; each demo's task (batch seed, env slot, hole pose) is replayable.
- Round 0: visible-hole transformer BC on the demos, one deterministic rollout per demo task -> pool.
- Round r = 1..8: train the hole-hidden ICL transformer from scratch on the pool (concat pathway, d=128, L=2, 10k steps). Sample = (context from a pooled rollout, expert demo target); 1/8 of samples have no context. Context norm stats from the current pool. Eval on the 512 fixed eval tasks (same as `icl_fair_bc.py`), k=1..8, attempt 1 blind, attempt k sees attempt k-1. Then K=8 sliding-window rollouts on every demo task (and val task) are appended to the pool (successes included).
- Controls at rounds 1 and 8: another task's attempt-1 f with the exact goal (`randf`), attempt-1 context frozen (`fixed`).
- Pairing: `matched` = context from task t with t's demo; `random` = f/h from any pooled rollout, c recomputed for the target demo's goal.
- Context: `goal` [f+c], `fc` [f,c], `hist_c` [h x10, f, c]. `goal` has no f, so pairing is irrelevant; it is the null control for the loop.
- `--supervise dagger`: half of each batch is the student's own states with expert action labels.

## Pilot (local, fc, 100 demos, seed 1; `logs/runs/icl-op-pilot-fc-{m,r}-d100-s1`)
BC solves .77 of its own demo tasks but .117 of eval tasks, so most round-0 contexts are successes (c ~ 0). pass@8: matched .006 (round 1), .029 (round 2); random .020, .025, .012 (rounds 1-3). Val loss matched .124 -> .047 once on-policy data arrives. Grid therefore uses 8 rounds.

## Results
(filled in when the klone grid finishes)
