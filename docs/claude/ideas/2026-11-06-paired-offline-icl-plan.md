# Paired offline ICL on peg insertion (plan, 2026-11-06)

Status: PLAN, nothing implemented yet. Planned script: `examples/baselines/ppo/icl_paired.py` (reuses `Policy`, `rollout`, `make_ctx`, `step_feedback` from `icl_checker.py`, and `observed_hole_obs` from `peg_insertion_side_bias.py`).

## Goal
Take the BC policy trained on 500 expert demos (pass@1 ~0.35-0.4; checkpoint `logs/runs/offline-vis-ckpt-s1/checkpoints/bc-500.pt`, it sees the true hole pose) and use its failures as the in-context "previous attempt" for an offline ICL student that only sees a noisy hole pose. The student has to read the true hole pose out of the feedback on the failed attempt and fix its action.

## 1. Paired data: a configuration = (reset seed, env index, hole pose)
Everything random at reset is fixed by the pair (reset seed, env index) plus the hole pose `goals` passed through `reset(options=...)`: robot start qpos, peg pose, peg geometry (fixed per env index), box (hole) pose. Two rollouts that call `venv.reset(seed=sd, options=reset_options(goals))` with the same `sd` and `goals` start from an identical state (verified for PickCube and the peg env: max |initial obs diff| = 0 across repeats).
Collection loop, per batch of 512 configurations (`goals = full(N)`, seed `sd`):
1. Success: expert (true hole pose, `peg-expert-r1/ckpt_3576.pt`) from the reset (`sd`, `goals`). Stores states, expert actions, alive mask.
2. Failure: bc-500 policy (true hole pose, no context) from the identical reset (`sd`, `goals`).
3. Keep env i iff expert succeeds AND BC fails. Expect ~0.91 * 0.65 ~ 0.6 of configs, so ~1000 configs (2 batches) for 500 pairs; stop at 500 (collect more if we want N > 500).
4. Store per pair: `sd`, env index, goal (x, y, yaw), the success trajectory (states, expert labels, alive) and the failure trajectory (states, executed actions, per-step feedback). Assert after collection that re-resetting with the stored (`sd`, goals) reproduces the same initial obs for both rollouts (log max diff).
Not stored: the noise (below), because neither trajectory depends on it.
Caveat: GPU physics is not exactly repeatable (same-reset repeats flip outcome in ~3-5% of cases), so the stored trajectories are the ground truth; we never re-roll to "verify" a failure.

## 2. How the goal is hidden (noisy hole pose)
The student never observes the true hole pose. Its 43-d state is the clean 43-d obs with `box_hole_pose` (obs 35:42) replaced by `observed_hole_obs`: position + (dx, dy, 0), orientation rotated by dyaw about z (the existing `PegInsertionSideBias-v1` mechanism; `hole_offset` obs 43:46 is not part of the student obs, the hole radius 42 is peg radius + 3 mm so it leaks nothing).
- Noise eps = (dx, dy, dyaw), uniform in +-bias_xy / +-bias_yaw (env defaults 0.02 m / 0.1 rad via `PEG_OBS_BIAS_XY/YAW`; `--noise-scale` multiplies them and we sweep it).
- eps is fixed within a configuration (same for the query, the context states, and attempt 1 and 2 at test time), so the noise is a persistent bias the student must infer, not per-step jitter.
- Training: eps is resampled for every training sample (it is independent of the stored trajectories), a free augmentation. Eval: one eps per eval configuration, fixed for both attempts.
- The context states carry the same noisy hole pose as the query. The only channel with the true hole pose is the feedback (section 3).
- Alternative (already implemented in `icl_offline.py --hide-goal`): zero the pose and set a flag. Equivalent to noise-scale -> infinity; keep as `--goal-mode mask`.

## 3. How the feedback is generated
Same as the existing offline ICL (`step_feedback` in `icl_checker.py`), computed from the FAILURE trajectory's clean states (true hole pose), one vector per step:
`y_t = [hole_pos - peg_head_pos (3, world frame), peg_head_pos (3), cos(hole_yaw), sin(hole_yaw)]` (8-d).
So the true hole pose is a function of every feedback token (hole = y[:3] + y[3:6], yaw from cos/sin), and `hole - head` shows where the failed attempt ended up relative to it. Train: feedback of the BC failure on the same configuration. Eval: feedback of the student's own attempt 1 (clean states, true hole pose), subsampled identically.
Optional controls on the same feedback: `shuf_y` (feedback from another env), `no_ctx`, `no_y` (zeros).

## 4. Conditioning mechanism
The existing `Policy` (icl_checker.py): context = the previous attempt only, subsampled every 5 steps (`ctx_stride=5`, Tc=20 tokens).
- Token = Linear([normalised state (noisy-goal, 43-d incl. mask flag), executed action (8), normalised feedback y (8)]) + learned positional embedding over the 20 slots; a learnable null token is prepended.
- 2-layer pre-norm transformer encoder (d=128, 4 heads) over tokens -> memory. Key padding masks the steps after the context episode ended (and everything when there is no context).
- Query = MLP(normalised current noisy-goal state); 2 cross-attention + MLP blocks over the memory; linear head -> 8-d action (joint delta pos). No within-episode history: the only memory is the previous attempt.
- Training pairs: query states = the SUCCESS trajectory's states (noisy-goal version), target = expert action on the true goal at those states. Context = the paired FAILURE trajectory (same configuration) w.p. 0.5 (`ctx_prob`), empty otherwise, so attempt 1 (no context) is in-distribution. MSE on alive steps. 6000 grad steps, batch 64 trajectories, lr 3e-4 cosine, as in `icl_offline.py`.

## 5. Evaluation and baselines
- 512 fresh eval configurations (fixed eval seed, one eps each). pass@1 (no context), pass@2 = attempt 1 or 2 where attempt 2 is conditioned on attempt 1 (same configuration, same eps, same reset seed), attempt-2-only, plus controls `no_ctx`, `shuf_y`.
- Baselines: (a) BC-noisy: same success demos, no context, noisy-goal obs (what the student can do without feedback), (b) BC-clean: the bc-500 policy itself (true goal, ~0.35) and the expert (0.91) as references, (c) oracle: student trained and evaluated with the true goal.
- Sizes N = 100, 200, 500 (nested prefixes), 3 seeds (seed changes the collection seeds and the init). Then a spatial / failure-mode analysis as for the BC policy.

## Known risks
- Train/test context mismatch: train contexts are BC failures acting on the TRUE goal; eval contexts are the student's own attempts acting on the NOISY goal (we saw this limit hidden-goal ICL before). Mitigation if needed: a second round that adds the student's own attempt-1 failures (expert partner from the same configuration) to the dataset (DAgger-style).
- Noise scale: too small and the noisy-goal student is already near BC (nothing to correct); too large and attempt 1 is hopeless. Sweep 1x / 2.5x / mask.
- 500 pairs is small for this task (BC-500 alone gets 0.35); the benefit of the context may be visible only relative to BC-noisy at the same N.
- Same-reset retries: even with identical resets outcomes can differ, so pass@2 without context (`no_ctx`) is the fair control for "just retrying".

## Interpretation to confirm
"Noise goal" is read as the noisy (biased) hole pose of `PegInsertionSideBias`, with the true pose recoverable only through feedback. If the intent is a fully masked goal, set `--goal-mode mask`; the rest of the plan is unchanged.
