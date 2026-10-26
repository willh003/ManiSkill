# PegInsertionSideBias-v1 and the peg task in `icl_episodic.py`

Env: `mani_skill/envs/tasks/tabletop/peg_insertion_side_bias.py` (subclass of PegInsertionSide-v1, 100 steps). State obs = the clean 43-dim obs
(qpos 0:9, qvel 9:18, tcp pose 18:25, peg pose 25:32, peg half size 32:35, box_hole_pose 35:42, hole radius 42) + `hole_offset` (3) = 46.
Reset options keep a task fixed across its episodes:
- `offset` (n,3) = (dx, dy, dyaw) bias of the observed hole pose (default random, +-2 cm, +-0.1 rad; env vars `PEG_OBS_BIAS_XY/YAW`)
- `box_pose` (n,3) = true box (x, y, yaw) (default: PegInsertionSide-v1 random spawn; ranges `BOX_XY_LOW/HIGH`, `BOX_YAW_*`)

Student task `PegTask` (`icl_episodic.py --env-id PegInsertionSideBias-v1`):
- biased (default): student sees `observed_hole_obs` (box_hole_pose shifted by the offset); task variable = offset.
- hidden (`--hide-hole`): box_hole_pose removed from the student obs (36 dims); task variable = box (x, y, yaw), fixed per task.
- feedback `final_error` (8 dims): [hole - peg head (world), peg head (world), cos/sin hole yaw] at the end of the episode, privileged. The hole is
  kinematic and does not move, so this is the hidden variable exactly (unlike the pick cube, which is pushed). `hints` / `success` as for pick.
- Prior expert (data diversity, reference): biased = acts on the unbiased obs; hidden = told the hole is at the default box pose (0, 0.3, yaw pi/2).
Caveats: geometry (peg/hole size, hole offset inside the box) is fixed per env index (reconfiguration_freq=0), so the same 512 geometries repeat every
iteration and eval; the student sees peg size/radius so it could in principle identify them. Expert: `logs/runs/peg-expert-fast/final_ckpt.pt` (success_once ~0.73).
Slurm: `examples/baselines/ppo/slurm/icl_peg.sbatch` (FB, SUFFIX, EXTRA=--hide-hole).
