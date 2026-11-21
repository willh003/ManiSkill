# PPO on train holes vs recurrent PPO warm-started from it (2026-11-21, in progress)

Tracker experiment `train-ppo-vs-recurrent`. 18 runs: PPO x {4, 8 holes} x 3 seeds (`pptr-M{M}-s{s}`), recurrent PPO x {4, 8 holes} x {clean, 4 cm noisy feedback} x 3 seeds
(`rl2tr-M{M}-{clean|n4cm}-s{s}`).

## Design
- Seed s uses `pool_seed = s` (holes = `checker_sample(M, 'all', cuda Generator(10000 + s), 'wide_corner')`, shared peg from `RandomState(1234 + s)`), so PPO and recurrent runs of a seed see
  the *same* holes and peg; the 4-hole set of a seed is the first 4 of its 8-hole set. Hole plot: `logs/renders/seed-sweep/task_sets_seeds.png` (`scripts/plot_task_sets_seeds.py`).
- PPO: `ppo_tasks.py` ppo_fast recipe, 2048 envs, `--near_hole_prob 0.25` default, 40M steps (earlier 60M runs were OOM-killed at ~46M), `--fixed --shared_peg --pool_seed s`.
- Recurrent PPO: `rl2_peg.py --n-ep 8 --fixed-tasks M --shared-peg --pool-seed s`, BC (DAgger, 12 iters) with the **PPO-on-train final ckpt as the expert** (not the full-workspace peg-expert-r1), then 300 PPO iters.
  clean = no feedback noise, noisy = 4 cm xy / 0.2 rad yaw. Caveat: the PPO expert is only trustworthy on its train holes; ep-1 labels use the prior-mean hole and the oracle rollouts use a believed
  (estimated) hole, both off the train holes.
- Eval protocol (identical for all): MetaPeg shared-peg env, deterministic, reset seed 12345, 1024 envs (first 512 reported); train = exactly pool[arange % M] rows; held-out = 512 holes
  (Generator 777). PPO ckpts evaluated with `dagger_peg.py --eval-only` (`evalpptr-M*-s*`); recurrent via rl2_peg eval (same code path / holes / reset seed).
  `ppo_tasks.py`'s own heldout/fullspace evals now also use the shared-peg table (before 2026-11-21 they used a random peg, see task-count-sweep correction).

## Results
TODO (jobs chained with `--dependency=afterok` on the PPO runs).
