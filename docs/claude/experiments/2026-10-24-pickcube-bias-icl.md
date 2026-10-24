# ICL student on PickCubeBias-v1: hidden observation bias on the cube position (2026-10-24)

Question: does the episodic ICL student (see `2026-10-23-icl-feedback-types.md`) also work on a manipulation task, where the
hidden quantity is a per-task bias in the observed cube position?

## Setup
- Env `PickCubeBias-v1` (`mani_skill/envs/tasks/tabletop/pick_cube_bias.py`): PickCube-v1 plus a per-task xy offset, uniform
  +-5 cm (`PICK_CUBE_OBS_BIAS`). The student sees cube-dependent obs entries (obj_pose, tcp_to_obj_pos, obj_to_goal_pos) shifted
  by the offset; the expert sees the clean obs. The offset is fixed across the k episodes of a task (`reset(options=dict(offset=...))`).
- Expert: PickCube-v1 PPO (`slurm/expert_pickcube_ppo.sbatch`), stopped early at ~13M steps (eval success_once 1.0);
  `logs/runs/pick-expert/final_ckpt.pt` is a copy of `ckpt_261.pt`. With the true cube position it gets 0.986 success in 16 steps.
  The same expert acting on the biased obs is a weak reference (0.14, but see the eval-seed issue below).
- Student: `icl_episodic.py --env-id PickCubeBias-v1` (now task-generic: `PushTTask` / `PickBiasTask`), 512 tasks/iter, 30 iterations,
  k=4, 50 steps. Slurm: `slurm/icl_pick.sbatch` (`FB=final_error|hints|success`). Final runs: `logs/runs/icl-pick-<fb>-v2/`.
- Feedback: `final_error` = true tcp->cube vector at the end of the episode; `hints` = 5 (state, expert action) pairs per
  episode; `success` = 0/1 flag.
- Plot: `logs/renders/icl-pick/feedback_comparison.png` (success_once vs training tasks, panels h=0, h=1, h=4).

## Results (final eval, success_once by number of previous episodes h)
| feedback | h=0 | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|---|
| true tcp->cube error | 0.30 | 0.97 | 0.99 | 0.99 | 0.99 |
| expert-action hints | 0.27 | 0.98 | 1.00 | 0.98 | 0.99 |
| success flag | 0.29 | 0.71 | 0.82 | 0.81 | 0.82 |

Controls at the end of training (success_once at h=1 / h=4):
| | shuffled feedback | no feedback | no context |
|---|---|---|---|
| true error | 0.14 / 0.14 | 0.26 / 0.23 | 0.28 / 0.30 |
| hints | 0.13 / 0.13 | 0.05 / 0.05 | 0.26 / 0.27 |
| success flag | 0.70 / 0.87 | 0.70 / 0.84 | 0.27 / 0.30 |

## Takeaways
- Error and hints recover the bias almost perfectly after one episode (0.97-0.98 vs 0.99 expert); hints learn fastest. The
  controls are at or below the no-context level, so the gain comes from the feedback content.
- Success flag: gain at h>=1 is **not** from the flag. Shuffled and no-feedback controls match the normal run (0.70-0.87), only
  removing the context steps drops it to ~0.27. The student infers the bias from its own past trajectories; the 1-bit flag adds
  nothing. (The first, buggy-eval run of this appeared to get 0.90 at h=1; the v2 run is the reference.)
- h=0 (~0.28) is the no-information level: same as the no-context control at h>=1.

## Bug found: eval-seed leak (fixed)
First runs (`icl-pick-<fb>`, superseded, not used) showed h=0 = 0.73-0.76 vs ~0.27 for identical inputs at h>=1. Cause: the eval
goals were sampled under `torch.manual_seed(eval_seed)` and the first eval episode reset the env with the same seed, so both
drew from the same random stream and the offset was exactly 0.5 x the cube xy (correlation 1.0) in that episode only. Fix
(`icl_episodic.py`): sample eval goals under `eval_seed + 1000`. Training was not affected (goals come from the global RNG,
not under a reset seed). Not re-run: PushT (h=0 was at chance there, so no sign of the leak); the "expert told default" reference
(0.14) in the first runs used the leaking seed and should be re-read from the v2 `results.json`.

## Why does the success flag help? (follow-up)
Eval-time zeroing of context features (success-flag student, h=1 / h=4; normal 0.71 / 0.86): observed cube entries 0.17 / 0.21,
qpos+qvel 0.49 / 0.62, tcp pose 0.59 / 0.75, actions 0.64 / 0.79, is_grasped 0.69 / 0.79, success flag 0.72 / 0.85. (OOD zeroing, so
read as "which features matter".) At the first grasped step, observed-cube minus tcp recovers the offset (R2 0.69, 1.7 cm) in the
53% of no-context episodes that grasp; the contact-based estimate (first step the observed cube moves) has R2 0.35.

Retrain with the grasp signal hidden from the student's obs (`--hide-grasp`, success-flag feedback, `logs/runs/icl-pick-success-hide{flag,fingers}`):
| hidden | h=0 | h=1 | h=2 | h=3 | h=4 | no-feedback control h=1 |
|---|---|---|---|---|---|---|
| nothing (v2) | 0.29 | 0.71 | 0.82 | 0.81 | 0.82 | 0.70 |
| is_grasped flag | 0.27 | 0.70 | 0.80 | 0.83 | 0.82 | 0.65 |
| flag + finger qpos/qvel | 0.27 | 0.63 | 0.73 | 0.75 | 0.78 | 0.62 |
No-context control stays ~0.25-0.30. **Hiding the grasp barely changes learning**, so the student is not mainly using the grasp event
(or the finger width). It still needs the context trajectories. Remaining candidates: tcp vs observed-cube relation during contact/pushing,
controller tracking error (qpos/qvel/actions). Next test: restrict the context to tcp pose + observed cube only.

Context restricted to tcp pose + observed cube xyz (`--ctx-features kin`: no joints, velocities or actions in the past-episode steps; the
current-state query still sees the full obs). Success-flag feedback, `logs/runs/icl-pick-success-kin{none,flag}` (flag = is_grasped also hidden):
| context / hidden | h=0 | h=1 | h=2 | h=3 | h=4 | no-feedback h=1 | no-ctx h=1 |
|---|---|---|---|---|---|---|---|
| kin context | 0.30 | 0.71 | 0.85 | 0.89 | 0.86 | 0.69 | 0.29 |
| kin context, is_grasped hidden | 0.31 | 0.75 | 0.85 | 0.88 | 0.89 | 0.70 | 0.30 |
**Controller error is not needed**: with only gripper pose and observed cube position in the context the student learns as well as with the
full context (0.71-0.75 at h=1 vs 0.71), and hiding the grasp flag as well changes nothing. The bias is inferred from the gripper trajectory
relative to the observed cube (e.g. where the gripper goes / stops, and how the observed cube moves on contact). Not tested: which part of
that trajectory matters (final positions only, contact events).

### Is the inference real? (diagnostics on `icl-pick-success-v2`, run locally with scratch scripts)
- Split of h=1 success by what happened in episode 0: cube lifted >=3 steps 0.86 (n=546); never lifted 0.63 (n=990); failed without a lift
  0.62 (n=954). No-context level is ~0.28, so carrying the cube (observed cube moves with the gripper at a fixed offset) is the strongest
  cue but not the only one.
- Counterfactual shift: add d to the cube x entries of the *context only* (as if the past episode's offset were o+d). h=1 success:
  d=-4/-2/0/+2/+4 cm -> 0.20/0.52/0.71/0.41/0.12; among episodes that still grasp, the grasp x relative to the true cube moves opposite to d
  (+0.8 / +0.6 / -0.2 / -1.1 / -1.3 cm). So the student causally uses the context's gripper-vs-observed-cube relation to estimate the offset
  (the effect is smaller than d and counted only on survivors). Not luck: 512-1000 tasks per cell, controls consistent across runs.
- Open: which events beyond a carry give information when nothing was lifted (pushing / near misses).
- Which previous-episode events carry information (`icl-pick-success-v2`, 2048 tasks, h=1 success by what episode 0 did; "pushed" = observed cube
  moved >1 cm, "lifted" = >=3 steps >1 cm above the table):
  | episode 0 | n | h=1 success | mean offset |
  |---|---|---|---|
  | lifted | 708 | 0.87 | 3.2 cm |
  | not lifted, cube pushed | 1234 | 0.65 | 4.1 cm |
  | not lifted, cube untouched | 106 | 0.32 | 4.9 cm |
  Controlling for |offset| (0-3 / 3-5 / 5-8 cm): lifted 0.86/0.89/0.84, pushed 0.64/0.68/0.58, untouched 0.75 (n=4)/0.29/0.32. Untouched is at the no-context
  level (0.28) whatever the closest approach (n too small to resolve it). So: **contact is what carries information** - a carry gives the most, a push still
  gives a lot, a miss gives nothing. Correlational (the shift test above is the causal one). 95% of episodes push the cube, which is why the success
  flag's gain does not depend on the flag: the information is in the contact geometry (gripper position when the observed cube starts to move,
  and how it moves).
