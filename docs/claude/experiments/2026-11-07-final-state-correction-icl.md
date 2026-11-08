# Final-state-correction ICL with MLPs (2026-11-07, seed 1)
Script `examples/baselines/ppo/icl_final_state.py` (plot `icl_final_state_plot.py`), run `logs/runs/icl-final-state` (results.json, ckpts `icl-N*.pt`, `bc-visible.pt`, `rollouts.pt`), log `logs/icl-final-state.log`, plot `logs/runs/icl-final-state/plots/success_vs_N.png` (copy in `logs/renders/final-state/`).

Setup: 500 expert demos (uniform configs) -> visible-goal MLP BC (3x512). BC rolled out from N fresh uniform configs; keep each final state (hole masked). Each rollout is paired with a RANDOM expert demo, resampled every batch (`--resample`, default): correction = expert final peg pose (7-d) - BC final peg pose; MLP input = [masked state, has_ctx, norm(final state 43 + correction 7)], target = expert action at every step of the paired demo (goal hidden; recoverable only as final peg + correction). Context dropped w.p. 0.5 in training; no BC regulariser. Eval (512 fresh configs): attempt 1 hidden-goal, no ctx; attempt 2 ctx = attempt-1 final state + (expert final peg on the same config - attempt-1 final peg).

Eval references: expert 0.896, visible-goal BC (500 demos) 0.459 (it solves 0.465 of its own 30000 uniform rollouts).

| N | pass@1 | pass@2 | attempt 2 | no ctx / shuffled pass@2 | train loss |
|---|---|---|---|---|---|
| 100 | 0.012 | 0.047 | 0.035 | 0.014 / 0.014 | 0.00535 |
| 300 | 0.004 | 0.037 | 0.033 | 0.006 / 0.018 | 0.00534 |
| 1000 | 0.006 | 0.041 | 0.035 | 0.006 / 0.010 | 0.00534 |
| 3000 | 0.002 | 0.039 | 0.037 | 0.002 / 0.014 | 0.00458 |
| 10000 | 0.010 | 0.037 | 0.029 | 0.018 / 0.021 | 0.00523 |
| 30000 | 0.008 | 0.025 | 0.018 | 0.012 / 0.016 | 0.00607 |

Flat ~0.04, barely above controls, far below BC 0.46: the MLP does not learn to use final peg + correction as the goal (train loss ~0.005 stays above the visible BC's 0.0016). Not diagnosed yet (ideas: feed final_peg + correction explicitly; drop the 43-d final state from the context; oracle single-token check on this MLP).

## Follow-up: explicit final pose + correction, no final state, context dropout fixed (2026-11-07)
New `--ctx-kind` options in `icl_final_state.py`: `pose` = context [attempt-1 final peg pose (7), correction = goal peg pose - final peg pose (7)] (raw final state removed); `goal` = [expert final peg pose] only; `hole` = [exact hole pose] only (references with the same MLP/training). Runs: `logs/runs/icl-final-{pose,goal,hole}[-nodrop]`, logs `logs/icl-final-*.log`, plot `logs/runs/icl-final-pose-nodrop/plots/success_vs_N.png`.

**Context dropout (p=0.5) broke everything**: with it, even `hole` (exact hole pose as context) got 0.031 pass@2 (train loss 0.0045); with `--ctx-drop 0` it gets 0.484 (loss 0.0017; visible-goal BC 0.459). Cause not understood (guess: unsolvable no-context samples dominate the clipped gradient). Earlier flat results (state-ctx run, pose and goal with dropout) are therefore not informative about the idea. With ctx-drop 0 attempt 1 sees an empty (untrained) context; pass@1 ~0.005 regardless.

References (ctx-drop 0): exact hole pose only 0.484; goal peg pose only (expert final peg pose) 0.346; visible-goal BC 0.459; expert 0.896.

Pose-context student (ctx-drop 0), pass@2 / attempt 2 / no-ctx / shuffled:
| N | pass@2 | attempt 2 | no ctx / shuf | train loss |
|---|---|---|---|---|
| 100 | 0.420 | 0.414 | 0.008 / 0.008 | 0.00219 |
| 300 | 0.402 | 0.400 | 0.006 / 0.004 | 0.00213 |
| 1000 | 0.395 | 0.391 | 0.004 / 0.006 | 0.00219 |
| 3000 | 0.316 | 0.316 | 0.004 / 0.006 | 0.00234 |
| 10000 | 0.244 | 0.238 | 0.010 / 0.020 | 0.00298 |
| 30000 | 0.135 | 0.129 | 0.008 / 0.016 | 0.00310 |

Works at small N (0.42, above the goal-peg-pose-only reference 0.35) but gets WORSE with more rollouts, and train loss rises with N. Hypothesis: with few distinct final poses the MLP memorises them and uses the correction alone; with many it must learn the exact sum final+correction within 6000 steps and underfits (not tested). Next: add the explicit sum (goal pose) as an input, or train longer for large N.
