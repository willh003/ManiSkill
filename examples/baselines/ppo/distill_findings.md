# PushT-v1: distilling a recurrent student under noisy T-pose observations

Findings from the experiments run in this directory. Everything below was measured on one seed unless stated otherwise;
single-run differences of a few points are within noise (see "Measurement noise").

## Setup

- **Task:** `PushT-v1` (panda_stick, `pd_joint_delta_pos`), 100-step episodes, state observation (31 dims: 7 qpos, 7 qvel,
  7 tcp pose, 3 goal position, 7 T pose). Dense reward (`normalized_dense`, per-step reward in about [0, 1]).
- **Expert:** state-based PPO (`ppo.py`, 1024 envs, 25M steps, γ=0.99, MLP 3x256 tanh), `runs/pusht-ppo-state/final_ckpt.pt`.
  On the true state it gets `success_once` 0.93–0.96 and `success_at_end` 0.87–0.89 (the eval set size sets the range).
- **Noise:** only the *observed* T pose (the last 7 obs entries) is corrupted by a per-episode offset (dx, dy, dtheta), uniform in
  ±5 cm and ±45° at "scale 1"; the simulator, reward and success check use the true T (`eval_obs_noise.py`).
- **Student:** GRU policy (`distill_recurrent.py`) whose input at step t is `[noisy_obs_t, a_{t-1}, r_{t-1}]`, i.e. it
  conditions on the history of states, actions and rewards. Trained DAgger-style: roll out the student for k steps under the
  noisy observation, let the expert (which sees the true state) act for the rest of the episode, aggregate, retrain.
  k per episode is uniform on [lo·100, hi·100] with hi = 0.5, 0.8, then 0.9 from iteration 3 on.
- **Best recipe:** expert-action labels on *every* step ("relabel"), k allowed to start at step 1, 1024 episodes and 3000
  gradient steps per iteration, cosine LR decay, 20 iterations: 2.05M collected transitions (8% of the expert's 25M env steps),
  Adam lr 5e-4, batch 128 sequences, backprop through full 100-step sequences. Evaluation: the student alone, deterministic,
  on 1024 fixed episodes each iteration.
- **Metrics:** `success_once` (success at any step) and `success_at_end` (success at the final step). The student's
  `success_at_end` is always lower than `success_once`, as it is for the expert (0.87 vs 0.96): episodes do not end at success,
  so some episodes push the T back out afterwards.

## 1. The expert is very sensitive to observation noise

Expert acting on the corrupted observation, 64 episodes, success_once / success_at_end:

| Noise scale | Offset (±) | success_once | success_at_end |
|---|---|---|---|
| 0 | none | 1.000 | 0.859 |
| 0.25 | 1.2 cm, 11° | 0.375 | 0.125 |
| 0.5 | 2.5 cm, 22° | 0.047 | 0.000 |
| 1.0 | 5 cm, 45° | 0.016 | 0.000 |

At full noise (1024 episodes) the expert gets 0.025 / 0.003.

## 2. Main results at full noise (±5 cm, ±45°), 1024 eval episodes

| Policy | Sees | success_once | success_at_end |
|---|---|---|---|
| Expert on true state | true T pose | 0.960 | 0.886 |
| Expert on noisy obs | noisy T pose | 0.025 | 0.003 |
| PPO, no memory, trained on noisy obs (50M steps) | noisy state only | 0.011 | 0.001 |
| Memoryless student (imitation, feed-forward) | noisy state only | 0.155 | 0.107 |
| Recurrent student, no reward | states + own actions | 0.452 | 0.355 |
| Recurrent student, with reward | states + actions + reward | 0.592 | 0.477 |
| Recurrent student + expert's previous action (privileged) | + expert action on the true state | 0.945 | 0.872 |

All students use the best recipe above and 20 DAgger iterations. Figure: `runs/distill_comparison.png`
(curves for the three recurrent runs, the expert levels and the PPO level as reference lines).
Per-run logs: `runs/pusht-distill-{noisyR6,noRew,expHist,memless}/log.json`; PPO: `runs/ppo-noisy-obs/`.

## 3. Findings

1. **What the student is trained on matters most.**
   - Training on the executed actions (the student's own actions for the first k steps, the expert's afterwards) failed:
     about 1% success after 5 iterations. The prefix is self-imitation, with no corrective signal, and at test time the student
     never sees the expert-driven history it was trained to continue from.
   - Taking the loss only on the expert-controlled steps (t >= k) learns, but slowly: about 0.07 after 30 iterations
     (256 episodes per iteration).
   - Labeling every step with the expert's action on the true state (DAgger) is best: 0.13 after 5 short iterations of the
     early setting, and 0.59 with more data and training per iteration.
2. **Every step needs supervision.** With the default curriculum the lowest k is 20 (rising to 40), so steps 0–19 never get a loss
   under the masked objective. Letting the handoff start at step 1 improved the noisy runs from about 0.07 to about 0.22
   and a no-noise sanity check from about 0.55 to about 0.8.
3. **The pipeline itself is sound.** With no noise, the recurrent student with relabeling reaches the expert's level
   (0.946 vs the expert's 0.949 on the same episodes) within 2–3 iterations.
4. **History carries a lot of information.** With no reward at all the recurrent student reaches 0.45, against 0.025 for the
   expert on the same observation and 0.155 for a memoryless imitation student. The reward adds about 14 points on top
   (0.45 to 0.59), a gap that holds across iterations 4–20.
5. **Memoryless policies do poorly under this noise.** PPO trained directly on the noisy observation (the best fixed strategy
   across the noise values) stays at about 1% success, although its eval return rises from 12 to 22 (it learns to push the T
   nearer the goal on average, not precisely). The imitation-trained memoryless policy reaches 15%.
6. **The information is sufficient, if given.** Adding the expert's action on the true state at the previous step brings
   the student to the expert's level from iteration 2 on. This signal is privileged and could be partly a copying shortcut
   (consecutive expert actions are similar), so it shows that the information is enough, not that the student infers the pose.
7. **The student plateaus at about 0.59, well short of 0.95.** From iteration 11 to 20 the curve is flat (0.52–0.59) and its
   action error against the expert is about 20 times the no-noise run's (0.021 vs 0.001). My reading (not tested) is that
   early in an episode the T's true pose is not identifiable from a single noisy state and the reward is a weak cue, so
   the student cannot match an expert that sees the pose.

## 4. Measurement noise and a note on the videos

- The same student re-evaluated on the same seed gave 0.607 and 0.609 `success_once`, but 114 of 1024 episodes (11%) flipped
  between the two runs (GPU physics is not fully deterministic). Treat numbers as ±1.5 points; with 256 episodes per point
  the error is about twice that.
- The first overlay video showed only 2 successes among its 8 episodes, while the eval rate was about 60%. This was chance
  (7 of 128 blocks of 8 episodes have 2 or fewer successes, about 5.5%), not a bug. `student_overlay_mix.mp4` in
  `runs/pusht-distill-noisyR6/` shows a mix (5 successes, 3 failures, chosen by episode index from one 64-episode rollout).

## 5. Caveats and open questions

- **Single seed** for everything. The reward effect (14 points) and the memoryless gap are large relative to the noise, but
  small differences (a few points) are not established.
- **The memoryless controls are not conclusive.** Neither knows the time step, which the recurrent student can count, so part
  of the gap could be timing and not information gathering. The imitation memoryless policy averages the expert's actions
  over the possible true poses. PPO's return was still rising slowly at 50M steps with hyperparameters tuned for the clean
  task, so 1% may understate what a memoryless policy can reach (it is below the imitation baseline).
  A memoryless policy with the step index as an input would separate timing from information gathering; I have not run it.
- **Not run:** a sweep over noise scales for the student (the expert's breakdown is in section 1), a larger network or
  longer history for the student, and multiple seeds.
- **Eval-set variation:** the first runs used 256 evaluation episodes, the later ones 1024; the expert baselines differ
  accordingly (0.930 vs 0.960 `success_once`).

## 6. Infrastructure notes

- `import torch` from the venv on `/gscratch/weirdlab` became very slow for a period (minutes, including from the login
  node), and several jobs appeared hung at startup. The sbatch scripts now do a preflight (imports + CUDA init under a
  timeout, retried up to 5 times). A stack dump (`faulthandler`) at the top of `distill_recurrent.py` shows where startup
  stalls. Consider reporting the slow reads to help@uw.edu.

## Code added in this directory

`eval_obs_noise.py` (observation-noise evaluation and overlay videos), `distill_recurrent.py` (recurrent student; flags
`--relabel`, `--k-lo-frac`, `--noise-scale`, `--no-reward`, `--expert-in-history`, `--memoryless`, `--lr-decay`),
`eval_student.py` (per-episode re-evaluation and mixed-episode videos), `ppo_noisy_obs.py` (memoryless PPO on the noisy
observation), `plot_distill.py` and `plot_compare.py` (figures).
