# Checkerboard hold-out test: ICL (per-step feedback) vs BC (peg task)

Code: `examples/baselines/ppo/icl_checker.py` (standalone; reuses `PegTask(full_info=True)`, `checker_sample` in `icl_tasks.py`), job `slurm/icl_checker.sbatch` (`MODE=icl|bc`).

- Goals: the box xy range (x in [-0.05, 0.05], y in [0.2, 0.4]) is split into 2x2 cells (i = x half, j = y half). White = (0,0), (1,1); black = (0,1), (1,0).
  Goal = xy uniform in a random cell of the region, yaw uniform in the usual range. Train on white; eval on white and black (512 fixed goals each).
- Student obs = true obs without the hole radius (42 dims) + a mask flag (43). Expert = strong peg expert (`peg-expert-r1`, 1.00 on both regions).
- ICL iteration (N=512 tasks, white goals): attempt 1 = expert acting on a random wrong hole pose (any cell, `wrong_expert_obs`), per-step feedback
  `step_feedback` = (hole - peg head, head, cos/sin hole yaw) with the TRUE hole; attempt 2 = student (hole pose masked) conditioned on attempt 1
  (every 5th step: state, executed action, feedback, one token each). Stored obs have the true pose; actions relabelled by the expert on the true pose.
  Training: each sample (query + context) has its hole pose zeroed + flag set with p=0.5 (`--pose-dropout`). Queries: attempt 1 with h=0, attempt 2 with h in {0,1}.
- BC: same network, no context, 2N clean expert demos on white per iteration, no masking.
- Eval: pass@1 = attempt 1 success_once, pass@2 = attempt 1 or 2. ICL attempt 2 is conditioned on attempt 1; BC (and ICL `no_ctx`) attempt 2 is an independent reset of the same goal.
  ICL controls at the end: `no_y`, `shuf_y`, `no_ctx`.
- Caveats: wrong poses range over black cells too (feedback displacements cover black ranges); ~5% of wrong-pose episodes succeed by accident; BC has no pose dropout.
