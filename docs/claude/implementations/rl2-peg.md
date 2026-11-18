# Recurrent PPO on multi-episode hidden-hole peg (`rl2_peg.py`, `meta_peg.py`)

Plan: `ideas/2026-11-17-rl2-peg-upper-bound-plan.md`. Results: `experiments/2026-11-17-rl2-peg-upper-bound.md`.

## Env: `examples/baselines/ppo/meta_peg.py` (`MetaPeg`)
- Wraps PegInsertionSideBias-v1 (state, pd_joint_delta_pos, normalized_dense reward, `max_episode_steps=1e6`, `auto_reset=False`) so
  resets are manual. One meta-episode = `n_ep` x 100 steps, all envs in lockstep, same box pose (task) in every episode
  (`box_pose` reset option, offset 0). `same_init=True` resets every episode with the same seed (same robot / peg start, as in the ICL evals).
- f = peg pose (obs 25:32) at the first success of the episode, else at step 100. Feedback c = hole pose (obs 35:42) - f (raw 7-dim difference,
  same as `icl_transformer.mk` with `--goal-ref hole`). Noise: c[:2] += N(0, fb_noise_xy); quaternion part replaced by (z-rotation by
  N(0, fb_noise_yaw)) * true hole quat - f quat. `f_hist` / `c_hist` keep all episodes' feedback.
- Actor obs (52 dims for 2 episodes): clean obs 0:35 (no hole pose / radius), episode one-hot, t/100, f, c, fb_valid. Holds the latest
  feedback until the next episode ends. Critic obs = actor obs + clean 43-dim obs (true hole). No reward input.
- Helpers: `load_expert` (ppo.Agent, 43-dim), `prior_hole_obs` (hole at the prior mean box x 0, y 0.3, yaw pi/2), `believed_hole_obs`.

## References: `meta_peg_oracle.py`
expert (true hole every episode), prior (prior-mean hole every episode), oracle (ep 1 prior, ep k>1 expert on mean_j(f_j + c_j)).
Outputs `logs/rl2-peg/oracle-<cfg>.json`.

## Training: `rl2_peg.py`
- Actor: obs norm -> 2x256 tanh -> GRU(256); head MLP on [enc, gru] (skip path) -> action mean; state-independent logstd.
  Critic: obs norm -> 3x256 tanh MLP on the critic obs (privileged, feedforward).
- BC (DAgger, `bc_iters` x (1 meta-episode rollout of num_envs + `bc_steps` grad steps on full 200-step sequences, aggregated data)):
  rollouts are a per-env beta-mix of student and scripted oracle (beta 1 -> 0 over `bc_beta_iters`); labels = expert on the true hole,
  except episode 1 with `--ep1-label prior` (default) = prior-mean-hole expert. Obs norms are set from the first (oracle) batch. Saves `bc.pt`.
- PPO: one rollout = one meta-episode; GAE gamma .99 / lambda .95, terminal only at the meta-episode end (credit flows across episodes);
  minibatches = groups of envs with full sequences from h=0; logstd reset to `init_logstd` (-1.5) after BC; first `value_warmup` iterations
  update only the critic; optional `bc_coef` auxiliary loss (off by default); target-KL early stop .05.
- Eval: deterministic on the 512 fixed eval tasks (wide_corner 'all', eval_seed 12345), tiled to num_envs; `succ_ep*` / `pass@n` = first 512,
  `all_*` = all envs. Final controls in results.json: `shuf` (other task's f, c in episodes >= 2), `zero` (f, c, valid zeroed), `hreset`
  (GRU state reset at each episode start).
- Resume: `checkpoints/latest.pt` (saved at every eval) or `bc.pt`. Checkpoints `it<k>.pt` every eval.
- Klone: `scripts/cluster/submit_cmd.sh rl2-peg-<cfg> "<3 seeds in parallel>"` (see the experiment note for the exact command).
- Tracker: `scripts/rl2_metrics.py OUT run_dir...` -> metrics / lastMetrics json for ArtifactData.
