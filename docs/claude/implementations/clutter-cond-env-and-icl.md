# Conditioned clutter pick env + ICL/BC (2026-11-17)

## Env `PickClutterYCBCond-v1`
`mani_skill/envs/tasks/tabletop/pick_clutter_ycb_cond.py` (the upstream `PickClutterYCB-v1` is a stub without reward/success; left untouched).
- Same K=4 YCB objects in every sub-scene: `077_rubiks_cube, 061_foam_brick, 011_banana, 056_tennis_ball` (`MODEL_IDS`). No reconfiguration needed.
- Layout: per-object (x, y, yaw), x in +-0.15, y in +-0.22, rejection-sampled so xy bounding circles don't overlap (`sample_layout(n, gen)`).
- Task: lift object `target`. success = target grasped and lifted >= 0.10 m; fail = any other object lifted >= 0.05 m. Both terminate. `info["lifted_id"]` (-1 = none). 75 steps.
- Reset options `layout` (n,K,3), `target` (n,). With `robot_init_qpos_noise=0` the start state is determined by (layout, target).
- State obs 75-d: qpos 0:9, qvel 9:18, tcp 18:25, obj_pose 25:53 (by id), obj_lift 53:57, obj_grasped 57:61, **target block 61:75** (onehot, target pose, tcp->target). `obs_layout()`, `TARGET_SLICE`. Students without the condition use obs[:61].
- Dense reward: reach target + grasp + 2*lift progress - non-target lift - 0.5*non-target grasp; 5 at success; normalized /5.
- Assets: `python -m mani_skill.utils.download_asset ycb` (done locally).

## Expert
`ppo_fast.py --env_id=PickClutterYCBCond-v1`, state, pd_joint_delta_pos. Local 5090 recipe (30 GB RAM box): 1024 envs, num-steps 32, 16 minibatches, no cudagraphs, `--no-capture-video` (eval video OOMs), run inside `systemd-run --user --scope -p MemoryMax=14G -p MemorySwapMax=0`. ~7.5 GB RSS. `ppo_fast.py` needed `@torch.no_grad()` on rollout/gae to run without cudagraphs.

## ICL vs BC: `examples/baselines/ppo/icl_clutter.py`
- Bank: per layout the expert is run for each of the K conditions; layout kept only if all K lift their object. Cached in `logs/clutter_bank/`.
- ICL sample: target t; random searcher history (random permutation tried until t, then t repeated); last 3 attempts as tokens [final obj xy (K*2), lifted flag per obj (K), feedback bit]; label = expert(t) action. No "right" token -> label uniform over untried objects -> GMM head (5 comps).
- BC: same `GMMPolicy`, no context, target block visible. `bc_guess`: BC with a random condition each attempt (memoryless guessing).
- Eval: 512 held-out layouts, 6 attempts; perf@k, pass@k, repeat (lifted an already-wrong object), nolift; ICL controls `none` (no context), `shuf_fb` (random feedback bits). References: oracle search, analytic guessing.
- Sampling: `--sample sticky` (keep GMM component while weight >= .2), `step`, `mean`.
- Runs: `logs/runs/icl-clutter-d<N>-s<seed>/` (results.json, checkpoints bc/icl(-best).pt, metrics.jsonl).
