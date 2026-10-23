# PushT with a hidden goal: expert PPO and an in-context-learning student (2026-10-23)

Question: if the goal T pose is hidden from the policy, can a transformer conditioned on previous episodes plus corrective
feedback recover it in context? Implementation: `docs/claude/implementations/pusht-goal-env-and-icl-student.md`.

## 1. Expert on PushTGoal-v1 (sees the true goal)
Goal randomized per episode around the PushT-v1 goal (+-3 cm, +-45 deg unless noted). `ppo.py`, 1024 envs, 100 steps, gamma 0.99,
8 update epochs, state obs, `pd_joint_delta_pos`. Slurm: `examples/baselines/ppo/slurm/expert_goal_ppo.sbatch`.

| Run | Flags / goal range | Result |
|---|---|---|
| first sweep (6 runs) | `--no-partial_reset`, various ranges | all poor (<=0.3 success_once). Control: original PushT-v1 with the same flags only 0.56 |
| `diag-goal-pr` | no randomization, default partial_reset, 25M | eval success_once 0.98 |
| `pr-small` / `pr-mid` / `pr-full` | +-1.5cm/20, +-2cm/30, +-3cm/45; default partial_reset, 75M | success_once 0.98-1.0 for all |

- **Lesson:** `--no-partial_reset` (episodes do not end at success) badly hurts PPO on PushT. Use the default `partial_reset`.
- **Expert used for the student:** `logs/runs/pr-full/final_ckpt.pt`. Curves: `logs/renders/pusht-goal/goal_expert_training.png`.
  Video of the expert: `logs/renders/pusht-goal/goal_expert.mp4`.

### Post-success drift (not reward hacking)
Without termination at eval the expert had success_at_end 0.28 (old PushT-v1 expert 0.90) despite success_once 0.98.
`expert_drift.py` (256 episodes, no termination), results in `logs/renders/pusht-goal/drift_*_expert.json`:

| after first success | new expert | old PushT-v1 expert |
|---|---|---|
| post-success steps still in success | 35% | 96% |
| exits from the goal per episode | 5.8 | 0.5 |
| action norm after success | 0.80 | 0.31 |

Cause: training terminates at success (`terminated = success`), so post-success states are never trained on; the policy
extrapolates to jittery actions there. In a non-terminating setting the success reward (1.0 normalized vs <=0.35 outside) makes
leaving the goal strictly worse, so it is not reward hacking at eval. Separately, shaping rewards hovering outside the goal
(~0.33/step) can exceed the one-off success bonus under termination; the `pr-*` logs show this tension early in training
(return ~30 at 29% success, falling to ~10 at 97% success as episodes shortened 82 -> 40 steps).
**Decision:** evaluate with the same termination as training. `ppo.py` eval now defaults to `--eval_partial_reset`.

## 2. ICL student (transformer over previous episodes + feedback)
`icl_episodic.py`, 512 tasks/iteration, 20 DAgger iterations, k=4 episodes, feedback = final T pose error
(dx, dy, cos dyaw, sin dyaw) computed at the termination step. Slurm: `slurm/icl_episodic.sbatch`.
Run dir: `logs/runs/icl-terminate/` (plots/icl_curve.png, results.json, checkpoints/policy_final.pt).

Eval, success_once / mean episode length by number of previous episodes h (fixed fresh goals):

| | h=0 | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|---|
| student | 0.03 / 98 | 0.93 / 42 | 0.96 / 40 | 0.95 / 41 | 0.95 / 41 |
| shuffled feedback | 0.04 | 0.03 | 0.02 | 0.02 | 0.02 |
| no feedback | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 |
| no context | 0.03 | 0.04 | 0.03 | 0.03 | 0.04 |

Expert with true goal: 0.99 / 37 steps. Expert told the default goal: 0.03 / 98.

**Takeaway:** one feedback episode recovers the goal (93%); the controls show the gain is from the feedback content.
No gain beyond h=1 because the feedback plus the observed final T pose determine the goal exactly. Needs weaker feedback
(scalar IoU, sparse hints) to make extra episodes matter. An earlier run without termination masking (`logs/runs/icl-full/`)
gave the same picture (0.03 -> 0.94 at h=1).
