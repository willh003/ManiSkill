# PushTGoal-v1 and the episodic ICL student

## Env: `mani_skill/envs/tasks/tabletop/push_t_goal.py`
- Subclass of PushT-v1 with a per-episode goal (x, y, yaw) sampled around the default goal. Range via env vars
  `PUSHT_GOAL_POS_RANGE` (m, default 0.03) and `PUSHT_GOAL_YAW_DEG` (default 45), so PPO sweeps need no new CLI.
- Obs (33 dims): qpos 0:7, qvel 7:14, tcp pose 14:21, goal pos 21:24, goal cos/sin 24:26, T pose 26:33.
- `reset(options=dict(goal=(N,3) tensor [dx,dy,dyaw]))` fixes the goal, which is how a multi-episode "task" is built.
- Per-env `world_to_goal_trans` makes the intersection/success per-goal. `tee_goal_error()` gives the privileged feedback.
- Checks: `examples/baselines/ppo/check_push_t_goal.py`, `compare_envs.py` (matches PushT-v1 reward/success when the range is 0).
- Termination is inherited: `terminated = success`.

## Student: `examples/baselines/ppo/icl_episodic.py`
- Policy: context encoder (transformer) over subsampled (s, a) steps of past episodes (`--ctx-stride 5`) plus one feedback token
  per episode and a learned null token; the current states cross-attend to this memory. The student never sees the goal.
- DAgger loop: k student episodes with growing context (iteration 0 and 25% of later slots use an expert told the default goal),
  one expert episode on the same goal, expert labels on every visited state; the expert episode is paired with every context c_h.
- Episodes terminate at first success; the vector env keeps stepping, everything after is masked (loss, context, feedback).
- Eval: fresh fixed goals, k+1 episodes; controls `shuf_y`, `no_y`, `no_ctx`. Feedback kinds: `--feedback final_error|success`.

## Run directories: `examples/baselines/ppo/run_logging.py`
`config.json`, `train.log`, `metrics.jsonl`, `tb/`, `checkpoints/`, `results.json`, `plots/` under `logs/runs/<name>/`.

## `ppo.py` changes (shared baseline)
`eval_partial_reset` now defaults to True (eval terminates like training), only each env's first episode is counted, and
`eval_reconfiguration_freq` defaults to None (= 1 unless partial reset). `success_at_end` is not recorded in that mode.
