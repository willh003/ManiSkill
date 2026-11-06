# PickCube BC vs number of expert demos (2026-11-06)

Code: `examples/baselines/ppo/pick_bc.py`. Run: `logs/runs/pick-bc/` (`data-s{1,2,3}.pt` = 200 successful expert demos per seed, `checkpoints/bc-{N}-s{seed}.pt`, `results.json`).

Setup: PickCube-v1, state obs (42-d), `pd_joint_delta_pos`, 50 steps. Expert = PPO `logs/runs/pick-expert/final_ckpt.pt` (0.988 on the 512 eval resets). Demos = successful expert episodes only, ending at first success (~16 steps). Nested prefixes N = 10, 50, 100, 200 demos. BC = MLP 3x256 ReLU, MSE to the expert action, 5000 grad steps, batch 256 frames, lr 3e-4 cosine, no context. Eval: 512 fixed resets, deterministic, success_once. 3 seeds (different demos).

| N demos | 10 | 50 | 100 | 200 |
|---|---|---|---|---|
| frames (~) | 190 | 830 | 1610 | 3270 |
| success, seeds 1/2/3 | 0.023/0.045/0.033 | 0.512/0.475/0.393 | 0.811/0.730/0.775 | 0.936/0.928/0.922 |
| mean | 0.034 | 0.46 | 0.77 | 0.93 |

Takeaway: smooth scaling, 10 demos is near chance, 200 demos is near the expert (0.99). Train loss ~0 everywhere (data coverage limits it, not optimisation).

## Failure modes and stochasticity (`pick_bc_analysis.py 50 100`; plots `logs/renders/pick-bc/analysis_N{50,100}.png`)
512 fixed eval resets, 3 policy seeds. Failure = never succeeds in 50 steps; categories from env state (is_grasping, cube-goal distance, tcp-cube distance), applied in the order listed.
| fraction of all episodes (mean of 3 seeds) | N=50 | N=100 |
|---|---|---|
| success | 0.45 | 0.77 |
| never reached cube (tcp > 5 cm) | 0.00 | 0.00 |
| reached, never grasped | 0.056 | 0.007 |
| grasped, then lost | 0.074 | 0.040 |
| held at end, cube not at goal | 0.40 | 0.17 |
| cube at goal, robot not static | 0.021 | 0.007 |
- Dominant failure: the cube is grasped and held but not brought to the goal (final cube-goal distance 0.05-0.35 m, median ~0.2 m, vs threshold 0.025). Grasping itself mostly works. Not separated: xy vs height error, or too slow vs wrong direction.
- Success drops mildly with the initial cube-goal xy distance (0.52 -> 0.32 across bins at N=50) and shows no clear pattern over initial cube xy.
- Stochasticity (same reset repeated 32x, since PickCube resets randomise cube and goal, there is no same-pose-new-reset to repeat): N=50: 42% of resets always succeed, 50% always fail, 8% mixed; N=100: 75% / 20% / 6%. Outcome is mostly determined by the reset. 64-68% of repeats never diverge in action; the rest diverge at step ~0-12 (GPU nondeterminism).

## Lift criterion (`pick_bc_lift_eval.py`; results `logs/runs/pick-bc/lift_results.json`, plot `logs/renders/pick-bc/lift_vs_demos.png`)
Same checkpoints/resets, success = cube max z over the 50 steps >= threshold (goal ignored; cube rests at z=0.02). Mean of 3 seeds:
| demos | lift >= 0.05 | >= 0.10 | >= 0.15 | >= 0.20 | reach goal (orig.) |
|---|---|---|---|---|---|
| 10 | 0.54 | 0.49 | 0.34 | 0.20 | 0.03 |
| 50 | 0.88 | 0.71 | 0.52 | 0.34 | 0.46 |
| 100 | 0.96 | 0.80 | 0.63 | 0.47 | 0.78 |
| 200 | 0.95 | 0.77 | 0.65 | 0.50 | 0.93 |
| expert | 0.91 | 0.72 | 0.56 | 0.40 | 0.99 |
Caveat: the demos/expert carry the cube to a random 3-D goal (goal height varies ~0-0.3 m), they do not lift to a fixed height. So the lift rate is capped by the goal height distribution (the expert itself only reaches 0.72 at 0.10 m), and BC at N>=100 beats the expert at low thresholds. A clean lift task needs a fixed-height target (new expert/demos) or conditioning on goal height.

## Goal-blind BC (`pick_bc_nogoal.py`; run `logs/runs/pick-bc-nogoal/`, ckpts `bc-nogoal-{N}-s{seed}.pt`, `results.json`)
Same demos (`pick-bc/data-s*.pt`), same MLP/steps, but goal_pos (obs 26:29) and obj_to_goal_pos (39:42) are removed from the student obs (36-d). Demos are still goal-directed. 512 eval resets, success = cube max z >= threshold. Mean of 3 seeds, with goal -> without goal:
| demos | lift>=0.05 | >=0.10 | >=0.15 | >=0.20 | reach goal |
|---|---|---|---|---|---|
| 10 | 0.54 -> 0.63 | 0.49 -> 0.53 | 0.34 -> 0.41 | 0.20 -> 0.30 | 0.03 -> 0.01 |
| 50 | 0.88 -> 0.94 | 0.71 -> 0.87 | 0.52 -> 0.70 | 0.34 -> 0.52 | 0.46 -> 0.01 |
| 100 | 0.96 -> 0.97 | 0.80 -> 0.89 | 0.63 -> 0.70 | 0.47 -> 0.49 | 0.78 -> 0.02 |
| 200 | 0.95 -> 0.98 | 0.77 -> 0.93 | 0.65 -> 0.82 | 0.50 -> 0.61 | 0.93 -> 0.02 |
Goal-blind lifts more reliably than the goal-conditioned policy (no longer pulled to low goals), median max z ~0.2 m at N>=50, but cannot reach the goal. Lift is measured as max z only (not held).
