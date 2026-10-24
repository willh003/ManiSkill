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
