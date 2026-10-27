# Peg hidden hole with the strong expert

Same as the hidden-hole runs in `2026-10-26-peg-hole-icl.md` (`--hide-hole`, student's own episodes, 30 iters, 512 tasks, k=4) but the expert is the
resumed PPO policy `peg-expert-r1` ckpt_3576 (copied to `runs/peg-expert-r1/strong_ckpt.pt` on klone; 0.99 success_once on 256 envs, 2 seeds;
r2 ckpt_3526 gives 0.99 too). Runs `logs/runs/icl-peg-{final_error,hints,success}-hidden-strong` (jobs 42415075/76/113).
Plot: `logs/renders/icl-peg/peg_strong.png` (`examples/baselines/ppo/plot_peg_strong.py`).

| feedback | h0 | h1 | h2 | h3 | h4 |
|---|---|---|---|---|---|
| true hole pose | .07 | 1.00 | .99 | .99 | 1.00 |
| hints | .05 | .85 | .95 | .97 | .96 |
| success flag | .06 | .03 | .03 | .03 | .04 |

Expert with true hole 1.00, told the default hole 0.05. Controls: shuffled/no feedback -> 0-0.02 for final_error and hints; no_ctx ~0.07 for all.

- True pose and hints reach expert level (old cap with the 0.74 expert was ~0.72); hints need 2-3 episodes.
- Success flag stays at chance: 1 bit with the student's own (hole-blind) trajectories carries no hole information. Compare `2026-10-27-peg-scripted-first-episodes.md`, where a checkpoint acting on the true hole in episode 0 made the flag work.
