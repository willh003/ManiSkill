# Offline BC vs ICL, whole workspace (2026-11-04)

Code: `examples/baselines/ppo/icl_offline.py` (reuses `Policy`, `rollout`, `make_ctx` from `icl_checker.py`). Runs: `logs/runs/offline-s{1,2,3}/results.json`. Plot: `logs/renders/offline/bc_vs_icl.png`.

## Setup
- Goals: whole wide workspace (x +-0.15, y 0.1..0.5, normal yaw; `wide_corner` ranges, region "all"). Expert (`peg-expert-r1`) solves 0.896 of the 512 fixed eval goals.
- Per seed: 200 goals where the expert succeeds (nested prefixes N = 10..200). Optimal = expert trajectory on the true goal. Suboptimal = expert acting on a random wrong goal (failures only, resampled until it fails), feedback = per-step (hole - head, head, cos/sin yaw) with the true hole.
- BC: N optimal trajectories, no context. ICL: same N optimal trajectories as targets, context = the paired suboptimal trajectory + feedback w.p. 0.5, empty otherwise. Student always sees the true goal; no masking, no online data.
- 3000 grad steps, batch 64 trajectories, lr 3e-4 cosine, same net/steps for both. 3 seeds, 512 eval goals.
- pass@2: ICL attempt 2 is conditioned on attempt 1 + feedback; BC attempt 2 is an independent reset.

## Results (mean of 3 seeds; pass@1 / pass@2; expert 0.896)
| N | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| BC | 0.000/0.001 | 0.000/0.001 | 0.004/0.007 | 0.019/0.031 | 0.071/0.105 |
| ICL | 0.000/0.000 | 0.000/0.000 | 0.006/0.007 | 0.013/0.017 | 0.073/0.097 |
| ICL no_ctx (pass@2) | | | 0.012 | 0.046 | 0.134 |

## Takeaways
- Neither method generalises at these sizes: the goal is 3-D (x, y, yaw) and 200 demos reach only ~7% pass@1 (loss ~0 = memorisation of the demo goals). The comparison is dominated by this regime, so it says little about ICL's retry benefit.
- ICL pass@1 ~ BC pass@1 (the context-dropout mixture does not hurt or help attempt 1). ICL pass@2 <= BC pass@2: the failure context does not help; replacing it with no context (`no_ctx`) gives a higher pass@2 than using it, i.e. the context slightly misleads.
- Why: the student already sees the true goal, and the context (a wrong-goal trajectory) is paired with a different target distribution only through the goal, so nothing forces it to be used. Pass@2 gains for BC/no_ctx come from reset noise on a deterministic policy.
- Next: more steps / larger N (1k-10k) or a narrower workspace so BC is non-trivial; check whether context helps once pass@1 is ~0.3-0.6; consider making the correction informative (e.g. optimal action residuals) or hiding the goal at test time.

## Convergence check (N=200, seed 1; `--log-every`, val = expert-action MSE on the eval goals, no context)
Train loss converges to ~0 (BC 0.0011 at 3k steps, 0.00006 at 12k; ICL 0.0016 -> 0.0001). Val loss is ~0.05 already at step 500 and rises monotonically (BC 0.049 -> 0.055 at 3k -> 0.061 at 12k; ICL 0.049 -> 0.053 -> 0.057): pure overfitting, a ~50x train/val gap. Success does not move with steps (BC 0.061/0.096 at 3k vs 0.064/0.111 at 12k; ICL 0.082/0.104 vs 0.062/0.104). So optimisation is not the bottleneck; data coverage is. Training longer does not help, early stopping would give at most a small gain. The sweep above used 3k steps without val logging.

## Larger N (N = 500..10000, 6000 steps, 3 seeds; runs `logs/runs/offline-big-s{1,2,3}`, plot `logs/renders/offline/bc_vs_icl_large.png`)
`icl_offline.py` now collects in batches of 512 goals (expert solves 0.91; ~10.2k pairs). Mean over 3 seeds, pass@1 / pass@2 (expert 0.896 single attempt):
| N | 500 | 1000 | 2000 | 5000 | 10000 |
|---|---|---|---|---|---|
| BC | 0.343/0.527 | 0.568/0.768 | 0.712/0.854 | 0.771/0.884 | 0.789/0.898 |
| ICL | 0.373/0.518 | 0.548/0.736 | 0.689/0.842 | 0.764/0.890 | 0.781/0.886 |
| ICL no_ctx (pass@2) | 0.579 | 0.759 | 0.842 | 0.881 | 0.893 |
| ICL shuf_y (pass@2) | 0.460 | 0.685 | 0.808 | 0.863 | 0.874 |
Seed std ~0.02-0.03.
- Both scale smoothly with N; BC pass@1 saturates near 0.79 (below the 0.90 expert). Val loss keeps falling with steps here (no overfitting at N >= 1000).
- ICL ~= BC on pass@1 and pass@2 at every N (differences within seed noise). Replacing the context with none gives the same or higher pass@2 than using it, and shuffling the feedback hurts, so ICL does read the context but it does not help: the context is slightly harmful at small N, neutral at large N.
- pass@2 gain over pass@1 is similar for BC (reset noise: 0.18 at N=500, 0.11 at 10k) and ICL; the failure context adds nothing beyond a retry. With the true goal visible, the wrong-goal trajectory + feedback carries no information the policy needs.

## Expert action in context (`--ctx-feedback expert_action`; runs `logs/runs/offline-ea-s{1,2,3}`, plot `logs/renders/offline/bc_vs_icl_expert_action_ctx.png`)
Same as above, but the per-step feedback in the context is replaced by the expert's action (true goal) at that context state (context token = state, executed action, expert action). Train: wrong-goal trajectory states; eval: attempt-1 states with the expert's label computed on them. 3 seeds, 6000 steps, N = 100..10000. pass@1 / pass@2 mean:
| N | 100 | 200 | 500 | 1000 | 2000 | 5000 | 10000 |
|---|---|---|---|---|---|---|---|
| BC | 0.021/0.034 | 0.057/0.104 | 0.344/0.533 | 0.572/0.768 | 0.713/0.857 | 0.776/0.890 | 0.786/0.898 |
| ICL | 0.025/0.031 | 0.091/0.135 | 0.370/0.516 | 0.544/0.727 | 0.689/0.831 | 0.766/0.887 | 0.769/0.893 |
| ICL no_ctx (p@2) | 0.052 | 0.160 | 0.562 | 0.757 | 0.831 | 0.883 | 0.889 |
| ICL shuf_y (p@2) | 0.027 | 0.122 | 0.493 | 0.715 | 0.834 | 0.887 | 0.893 |
| ICL attempt-2-only | 0.006 | 0.044 | 0.184 | 0.348 | 0.449 | 0.655 | 0.691 |
- No benefit: ICL ~= BC, and pass@2 with the context is <= pass@2 without it (no_ctx) at every N. Same picture as with the original feedback.
- Shuffling the expert actions barely changes pass@2 (0.893 vs 0.893 at 10k): the policy does not use them. Expected here: it sees the true goal, so it can compute the expert action itself, and the context is redundant.
- Attempt 2 alone (conditioned on attempt 1) is clearly worse than attempt 1 without context (0.69 vs 0.77 at 10k), so conditioning on the context *hurts*. Likely a train/test mismatch: train contexts are wrong-goal expert trajectories (all failures), eval contexts are the policy's own rollouts (often successes, ending at the first success). Not tested.

## Goal hidden at test time (`--hide-goal`, per-step distance-to-goal feedback; runs `logs/runs/offline-hide-feedback-s{1,2,3}`, plot `logs/renders/offline/bc_vs_icl_hidden_goal.png`)
The hole pose is masked (zeros + flag) in every query and in the context states, in training and eval, for BC and ICL. The only goal information is the context's per-step feedback (hole - head, head, cos/sin yaw) from the paired wrong-goal failure (train) / the policy's own attempt 1 (eval). 6000 steps, 3 seeds. (An expert-action-context hidden-goal run was started and killed on request; no results.)
pass@1 / pass@2 mean (expert 0.896 with the goal visible):
| N | 500 | 1000 | 2000 | 5000 | 10000 |
|---|---|---|---|---|---|
| BC | 0.009/0.018 | 0.009/0.023 | 0.016/0.030 | 0.014/0.028 | 0.012/0.023 |
| ICL | 0.006/0.027 | 0.011/0.095 | 0.010/0.206 | 0.016/0.299 | 0.008/0.326 |
| ICL no_ctx (p@2) | 0.012 | 0.025 | 0.022 | 0.025 | 0.020 |
| ICL shuf_y (p@2) | 0.012 | 0.019 | 0.015 | 0.020 | 0.019 |
- pass@1 ~1% for both (goal invisible, no information). BC pass@2 stays ~2-3% (a retry gives nothing).
- ICL pass@2 grows steadily with N (0.03 -> 0.33; seed std 0.01-0.04) and is >10x BC from N>=2000. Removing the context or shuffling the feedback drops it back to BC level, so the gain is from inferring the goal from the first attempt's feedback.
- Still far below the visible-goal level (0.89): one wrong-goal-style failure + feedback at N=10000 gets a third of the way; the curve has not saturated.
- Caveat: eval attempt-1 contexts are the policy's own (goal-blind) rollouts, not the wrong-goal expert trajectories seen in training; the train/test context mismatch (noted above) may be limiting the result.
