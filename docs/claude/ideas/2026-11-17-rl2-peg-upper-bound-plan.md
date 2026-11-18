# Plan: recurrent PPO (RL²-style) upper bound for hidden-hole peg with feedback (2026-11-17)

Goal: a recurrent PPO policy that solves PegInsertionSideBias-v1 with the hole hidden, over a meta-episode of 2 episodes on the same task,
with feedback c = hole pose − final peg pose f after episode 1. It is an upper bound for the ICL transformer
(`icl_onpolicy.py`, `--goal-ref hole`), since RL optimises the 2-episode return directly instead of imitating demos.
Later: noisy feedback, K > 2 episodes.

## Setting (matched to the ICL eval so the numbers compare)
- Task distribution: `checker_sample(n, "all", split="wide_corner")` box (x, y, yaw); eval = the same 512 fixed tasks (`eval_seed` 12345).
- Student obs: clean 43-dim obs with the hole pose (35:42) and radius (42) removed → 35 dims, plus inputs
  `[ep_idx one-hot, t/100, f (7), c (7), fb_valid]`. f, c are zero in episode 1 and held fixed through episode 2
  (same information as the ICL `fc` context). The GRU state carries across the boundary, so episode-1 history is also usable.
  No reward input (dense reward depends on the hole and would leak it).
- f = peg pose at the first success or at step 100 (`final_idx` convention); c = hole pose − f, 7-dim, raw quaternion difference (`mk`).
- Episodes in lockstep: one rollout = one 200-step meta-episode for all envs. Step 0: reset with new box poses. Step 100: reset with the same
  `box_pose` option (peg start re-randomised, geometry fixed per env slot since reconfiguration_freq=0). No partial resets needed.

## Step 0: reference numbers without RL (do first, ~1 h)
On the 512 eval tasks, with the strong expert `runs/peg-expert-r1/ckpt_3576.pt`:
1. expert with the true hole (ceiling for episode 2);
2. blind episode 1: prior expert (told the mean hole pose) and the hole-masked BC/ICL attempt-1 success (~.005);
3. scripted oracle: ep 1 = prior expert, ep 2 = expert acting on hole = f + c. With noiseless feedback this *is* the Bayes-optimal ep-2
   behaviour, so it's the target the RL policy should reach (pass@2 ≈ expert success). The RL run checks that the pipeline can learn it,
   and matters more once feedback is noisy (combining evidence, choosing informative ep-1 behaviour).

## Step 1: two-episode env wrapper (`meta_peg.py`)
`MetaPegVecEnv(num_envs, n_episodes=2, fb_noise=0.0)` around `ManiSkillVectorEnv(..., ignore_terminations=True)`:
- tracks success / first-success step per episode, f, builds c (+ optional Gaussian noise on position / yaw), builds the student obs;
- returns the privileged obs (true hole, ep_idx, t) for the critic;
- reward: env dense normalised reward per step, weighted by `w_ep[k]` (default 1 for both), plus an episode-success bonus.
Test: replay the scripted oracle through the wrapper and get the Step-0 numbers.

## Step 2: recurrent actor, privileged critic (`rl2_peg.py`, built from `ppo_fast.py`)
- Actor: MLP encoder (256) → GRU (256) → MLP head → Gaussian mean; state-independent logstd.
- Critic: feedforward MLP on [privileged full obs incl. true hole, ep_idx, t/100, f, c] (asymmetric actor-critic; no recurrence needed).
- Rollout: 1024–2048 envs × 200 steps; store the GRU state at step 0 (zeros), so PPO updates run BPTT over the full meta-episode, minibatching
  over envs (or 50-step chunks with stored states if memory is tight).
- GAE across the episode boundary (no `done` at step 100), so episode-1 actions get credit from episode 2. γ ≈ 0.99 (the expert used 0.97; with
  a 100-step episode 0.97 cannot carry credit across the boundary, which matters for the noisy case). Done only at step 200.

## Step 3: BC init (DAgger distillation from the expert)
Pretrain the actor before PPO so it starts with reasonable behaviour:
- labels = expert action on the true-hole obs at every visited state; rollouts by the student itself (DAgger, β-mix with the expert for the
  first iterations), conditioned on its actual f, c.
- Episode 2 labels are learnable exactly (hole = f + c). Episode 1 labels are not (hole hidden), so BC learns the posterior-mean action, i.e.
  aim at the prior mean hole; this gives some ep-1 success and a spread of f values.
- Check before PPO: ep-2 success near the scripted oracle; ep-1 near the blind baseline. If ep-2 BC is already ≈ oracle, PPO mostly needs to
  hold that and improve ep 1 (and the noisy case).
- Also pretrain the critic: a few PPO iterations with the actor frozen (value warm-up) to avoid destroying the BC policy with a random critic.

## Step 4: PPO fine-tune
From the BC init: logstd init ≈ −1.5, lr 1e-4 (actor) / 3e-4 (critic), clip 0.2, 4–8 epochs, target-KL early stop; optional BC/KL-to-init
penalty annealed to 0 over the first ~20% of training. Log per episode: success, success_once, pass@2, return, entropy, value loss, the
BC-label MSE on ep 2. Run 3 seeds. Local pilot on the 5090 (short), then klone L40 jobs (`scripts/cluster/`), each reported to the tracker.

## Step 5: eval and comparison
Deterministic eval on the 512 tasks: success ep 1, ep 2, pass@2, perf@2. Controls: ep-2 with shuffled c (another task's), with c = 0 / fb_valid
= 0, with the hidden state reset at the boundary (tests whether it uses the history vs only the explicit f, c). Compare against the scripted
oracle, expert, BC+noise and the ICL `fc` / `hist_c` runs (pass@2 from `logs/runs/icl-op-*`).

## Extensions (after 2 episodes noiseless works)
1. Noisy feedback: c + N(0, σ) on xy (σ = 0.5, 1, 2, 4 cm) and yaw; scripted Bayes reference = expert on the posterior mean.
2. K = 3, 4 episodes: feedback after each episode; the GRU must combine several noisy corrections (reference: posterior mean of all
   f_k + c_k). Here RL can beat the ICL policy most clearly.
3. Feedback that only gives partial info (e.g. direction / success flag) where ep-1 exploration matters.

## Risks
- PPO collapse of the BC init (value warm-up, low lr, KL penalty).
- BPTT over 200 steps: if unstable, truncate to 50-step chunks with stored states.
- Quaternion difference in c is not a clean rotation error; if the yaw part is hard to use, add (Δx, Δy, Δyaw) as a cleaner feedback variant
  and report both (the ICL runs used the raw 7-dim form).
