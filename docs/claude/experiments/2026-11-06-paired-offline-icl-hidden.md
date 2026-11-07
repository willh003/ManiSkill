# Paired offline ICL, hidden goal, seed 1 (2026-11-06)

Plan: `docs/claude/ideas/2026-11-06-paired-offline-icl-plan-v2.md`. Code: `examples/baselines/ppo/icl_paired.py`. Run: `logs/runs/icl-paired-s1/` (`data.pt` = 500 pairs, `checkpoints/bc-hidden.pt`, `icl-N{n}-stride{s}.pt`, `results.json`); stdout `logs/icl-paired-s1.out`. ONE seed only.

Setup: goal masked everywhere; hidden-goal BC on 500 expert demos; 500 pairs (expert success + hidden-BC failure, identical reset seed/goal, max initial-obs diff 0) from 1024 configs (expert solves 0.889, hidden BC 0.009); student initialised from BC, loss = context loss + 1.0 * BC-distillation regulariser; 6000 steps, batch 64; 9-d feedback incl. |tcp - peg|; eval 512 fresh configs, attempt 2 uses the same reset seed.

Reference: expert 0.900; hidden BC pass@1 0.008, pass@2 same reset 0.008, new reset 0.020.

| stride (tokens) | N | pass@1 | pass@2 | attempt 2 | no_ctx p@2 | shuf_y p@2 | no_y p@2 | final-head dist to BC rollout (median) |
|---|---|---|---|---|---|---|---|---|
| 5 (20) | 100 | 0.006 | 0.006 | 0.000 | 0.012 | 0.006 | 0.006 | 0.168 m |
| 5 | 200 | 0.010 | 0.020 | 0.010 | 0.016 | 0.016 | 0.010 | 0.164 |
| 5 | 500 | 0.006 | 0.035 | 0.029 | 0.006 | 0.020 | 0.006 | 0.145 |
| 10 (10) | 100 | 0.018 | 0.020 | 0.002 | 0.023 | 0.023 | 0.018 | 0.158 |
| 10 | 200 | 0.006 | 0.014 | 0.010 | 0.010 | 0.010 | 0.006 | 0.150 |
| 10 | 500 | 0.006 | 0.029 | 0.023 | 0.016 | 0.010 | 0.006 | 0.171 |
| 100 (1) | 100 | 0.008 | 0.014 | 0.012 | 0.010 | 0.012 | 0.008 | 0.159 |
| 100 | 200 | 0.012 | 0.016 | 0.012 | 0.018 | 0.014 | 0.012 | 0.162 |
| 100 | 500 | 0.014 | 0.020 | 0.012 | 0.020 | 0.018 | 0.014 | 0.163 |

Observations:
- Context gives almost nothing at N<=500: pass@2 0.014-0.035 (eval SE ~0.007). The only visible gap over the controls is stride 5 / 10 at N=500 (0.035 / 0.029 vs no_ctx 0.006 / 0.016), one seed, so suggestive at best. Last frame only: no signal.
- BC match is good per step (action MSE on BC states ~1e-3, falling with N) but poor in trajectory space: the student's final peg head ends a median 0.15-0.17 m from the BC rollout's on the same reset, within 1 cm for only 2-3% of episodes. Small action errors compound over 100 steps, so attempt-1 failures of the student are not the BC's failures in the sense of "same trajectory". The marginal outcome (pass@1 ~1% for both) does match. Not measured: how far two BC rollouts from the same reset drift apart (sim nondeterminism floor).
- Earlier unpaired hidden-goal ICL at N=500 also got pass@2 ~0.03; the paired data / extra feedback / matched contexts did not change that at this size.

## Sanity checks (is the low pass@2 a bug?)
1. Data (scratchpad `diag.py`, not in repo): the hole position recomputed from the stored feedback equals the true hole in the stored obs (max err 5e-8); both rollouts of a pair share hole pose, peg pose and geometry at t=0 (diff 0).
2. Teacher-forced expert-action MSE on success-trajectory states, context = paired failure (train pairs vs 467 fresh held-out pairs): train loss with the true context ~1e-3 (N=500) / 4e-5 (N=100) vs 0.025 without it, but held-out 0.032 (N=500) / 0.060 (N=100) vs 0.027 without context and 0.062 with a shuffled one. The student uses the context as a lookup key for training pairs and does not generalise (memorisation; a deterministic BC failure trajectory is a fingerprint of its configuration).
3. Oracle (`icl_paired_oracle.py`, run `logs/runs/icl-paired-oracle/results.json`): same 500 saved pairs and pipeline, but the context is ONE token holding only the true hole pose (position, cos/sin yaw; states/actions zero), queries masked. 512 eval configs, pass@1 with that context:
| N pairs | oracle-context student | shuffled ctx | no ctx | visible-goal BC (same trajectories, unmasked) |
|---|---|---|---|---|
| 100 | 0.021 | 0.000 | 0.000 | 0.016 |
| 200 | 0.102 | 0.004 | 0.000 | 0.098 |
| 500 | 0.311 | 0.006 | 0.000 | 0.344 |
The context path, masking, training loop and eval rollouts work: with the goal clean in the context the student matches visible-goal BC at every N and breaks without it. So the 0.03 of the trajectory-context student at N=500 is an extraction / memorisation problem, not a pipeline bug; ceiling at N=500 ~0.31-0.34.
Next: ablate what the context holds (hole-only -> + peg head / gripper distance -> + states and actions), more pairs, context dropout.
