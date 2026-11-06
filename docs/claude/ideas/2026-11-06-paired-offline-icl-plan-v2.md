# Paired offline ICL on peg insertion, hidden goal (plan v2, 2026-11-06)

Status: PLAN, nothing implemented. Supersedes `2026-11-06-paired-offline-icl-plan.md` (noisy goal -> hidden goal; failures now come from a hidden-goal BC policy; BC-matching regularizer; extra feedback; context sweep).
Planned script: `examples/baselines/ppo/icl_paired.py` (reuses `Policy`, `rollout`, `make_ctx`, `apply_mask`, `step_feedback` from `icl_checker.py`; `Y_DIM` becomes 9).

## Goal
An offline ICL student that never sees the hole pose. Attempt 1 (no context) must behave like a hidden-goal BC policy; attempt 2 sees the failed attempt 1 plus per-step feedback and has to recover the hole pose from it. Because the student's attempt 1 is (by construction) the BC policy, the contexts used in training (BC failures) and at test time (the student's own attempt 1) come from the same distribution.

## 0. The BC policy (assumption to confirm)
"BC policy on the masked goal" = BC trained from scratch on the N=500 expert demos with the hole pose masked in every input (same `Policy` class with an always-empty context, `icl_offline.py --hide-goal --methods bc --sizes 500`, 6000 steps). We train it (the existing `offline-hide-ckpt-*` runs saved only ICL checkpoints). Earlier hidden-goal BC pass@1 was ~1%, so nearly every configuration yields a failure.
The ~0.35-0.4 BC policy that sees the goal is no longer used for failures (it cannot be matched by a student with no goal). It stays as a reference. Alternative reading: feed the visible-goal BC policy masked inputs; I do not think that is intended (its behaviour on out-of-distribution masked inputs is arbitrary).

## 1. Paired data
Configuration = (reset seed `sd`, env index, hole pose `goals`): robot start qpos, peg pose, peg geometry (fixed per env index) and hole pose are all determined by `venv.reset(seed=sd, options=reset_options(goals))`; two resets with the same arguments give identical initial obs (max |diff| = 0, checked before).
Per batch of 512 configurations, same `sd` and `goals` for both rollouts:
1. Success: expert (true hole pose) -> states, expert actions, alive mask (stops at first success).
2. Failure: hidden-goal BC (masked inputs, no context) -> states, executed actions, expert labels at its states, per-step feedback. 100 steps, so the last frame is step 99.
3. Keep env i iff expert succeeds and BC fails (~0.9 of configs): 500 pairs from ~560 configs (2 batches). Store `sd`, env index, goal, both trajectories. Log the max initial-obs difference between the two rollouts of each pair (re-reset check).
4. We can afford more pairs (cheap); 500 is the primary setting, N in {100, 200, 500} nested, and extra sizes (2000, 5000) as a scaling sweep since earlier hidden-goal ICL only started to work at N >= 2000.

## 2. How the goal is hidden
Every student input has the hole pose (obs 35:42) zeroed and the mask flag set to 1 (`apply_mask`): the query states, the context states, in training and eval, for BC, for the ICL student and for the hidden-goal BC that generates failures. The goal never reaches the student except through the feedback y (below). The hole radius (obs 42) is peg radius + 3 mm, so it leaks nothing. Eval attempt 1 is the student's own rollout on masked inputs.

## 3. Feedback (per step, computed from the clean state of the failed attempt, true hole pose)
`y_t = [hole_pos - peg_head_pos (3, world frame), peg_head_pos (3), cos(hole_yaw), sin(hole_yaw), ||tcp_pos - peg_pos|| (1)]`, 9-d.
- First 8 entries: as before; the hole pose is a function of the token (hole = y[:3] + y[3:6], yaw from cos/sin), and `hole - head` says how far the attempt ended from the hole.
- New 9th entry: gripper-to-peg distance (obs tcp 18:21, peg center 25:28), says whether the peg is held and where it sits in the gripper. (Variant if scalar is too weak: add the 3-d vector tcp - peg.)
Train: feedback of the BC failure on the same configuration. Eval: feedback of the student's own attempt 1, built the same way.

## 4. Conditioning mechanism and context sweep
`Policy` as before: context tokens = Linear([normalised masked state (43), executed action (8), normalised feedback (9)]) + learned slot embedding, null token prepended, 2-layer pre-norm transformer encoder (d=128, 4 heads) -> memory with key padding mask; query = MLP(normalised masked current state); 2 cross-attention + MLP blocks over the memory; linear head -> action. The only memory is the previous attempt.
Sweep over which frames of the failed attempt form the context (`--ctx-stride`, `ctx_index` already supports all three):
| setting | stride | tokens |
|---|---|---|
| every 5 frames | 5 | 20 |
| every 10 frames | 10 | 10 |
| last frame only | 100 | 1 (step 99) |
All three use the same N, steps and seeds. In training the context is present w.p. `ctx_prob` = 0.5 and empty otherwise.

## 5. Training objective, with the BC-matching regularizer
Initialise the student from the hidden-goal BC weights (same architecture; with an empty context the cross-attention only sees the null token, so the student starts as exactly BC). Two losses per step:
1. Context loss (w.p. 0.5 samples): query = a state of the SUCCESS trajectory (masked), context = the paired FAILURE trajectory (masked, stride per setting), target = expert action on the true goal at that state. MSE over alive steps.
2. Regularizer (no-context samples): empty context, query = a state drawn from the failure trajectories (BC's on-policy states) and the success trajectories, target = the frozen hidden-goal BC's action on the same masked state. MSE, weight lambda = 1 (sweep 0.3 / 1 / 3 if the match is poor). This replaces the plain expert-imitation loss for empty contexts: with the goal hidden that loss only reproduces BC's mean-action behaviour anyway, and distilling the actual BC makes the match exact on the states that matter.
Result: the student with no context ~ BC on BC's own state distribution, so attempt-1 failures of the student ~ BC's failures (the train/test context mismatch is removed by construction).
Diagnostics for the match (eval configs, attempt 1): pass@1 student vs BC, per-step action MSE student vs BC along the BC rollout, and the distance between the final peg heads of the two rollouts (same reset, so near-identical trajectories expected up to sim nondeterminism, ~3-5% outcome flips).
Hyperparameters: 6000 grad steps, batch 64 trajectories, Adam lr 3e-4 cosine, 3 seeds (seed = collection seeds + init).

## 6. Evaluation and baselines
512 fresh configurations (fixed eval seed), hidden goal. pass@1, pass@2 (attempt 2 conditioned on attempt 1, same configuration and reset seed), attempt-2-only. Controls: `no_ctx` (same retry without context, the fair control for retry noise), `shuf_y` (feedback from another env), `no_y` (zeros). Baselines: hidden-goal BC (same data; pass@2 = an independent retry), the visible-goal BC at 500 (~0.35) and the expert (0.91) as references, and an oracle student trained and evaluated with the goal visible. Then the spatial / failure-mode / stochasticity analyses used for BC.

## Risks
- Hidden-goal BC is ~1% pass@1, so all gain must come from the context; with only 500 pairs the earlier hidden-goal ICL got pass@2 ~0.03. Expect to need the larger-N sweep; the last-frame-only setting tests whether the final error alone is enough.
- With one failure per configuration the student has no exposure to attempts that partly succeed; the regularizer pins attempt 1 to BC so eval contexts should look like training contexts, but covariate shift on attempt 2 (its own states) remains. Extension: also query on failure-trajectory states with the expert label (the labels are stored).
- The regularizer and the context loss compete for capacity/gradient; watch the BC-match diagnostics while training.
