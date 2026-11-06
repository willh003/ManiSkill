# PickCube BC vs number of expert demos (2026-11-06)

Code: `examples/baselines/ppo/pick_bc.py`. Run: `logs/runs/pick-bc/` (`data-s{1,2,3}.pt` = 200 successful expert demos per seed, `checkpoints/bc-{N}-s{seed}.pt`, `results.json`).

Setup: PickCube-v1, state obs (42-d), `pd_joint_delta_pos`, 50 steps. Expert = PPO `logs/runs/pick-expert/final_ckpt.pt` (0.988 on the 512 eval resets). Demos = successful expert episodes only, ending at first success (~16 steps). Nested prefixes N = 10, 50, 100, 200 demos. BC = MLP 3x256 ReLU, MSE to the expert action, 5000 grad steps, batch 256 frames, lr 3e-4 cosine, no context. Eval: 512 fixed resets, deterministic, success_once. 3 seeds (different demos).

| N demos | 10 | 50 | 100 | 200 |
|---|---|---|---|---|
| frames (~) | 190 | 830 | 1610 | 3270 |
| success, seeds 1/2/3 | 0.023/0.045/0.033 | 0.512/0.475/0.393 | 0.811/0.730/0.775 | 0.936/0.928/0.922 |
| mean | 0.034 | 0.46 | 0.77 | 0.93 |

Takeaway: smooth scaling, 10 demos is near chance, 200 demos is near the expert (0.99). Train loss ~0 everywhere (data coverage limits it, not optimisation).
