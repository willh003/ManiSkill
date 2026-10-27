# Peg: suboptimal checkpoint context episodes (full-info and hidden-hole)

Env `PegInsertionSideBias-v1`, 30 iters, 512 envs, weak expert (`peg-expert-fast`, success_once ~0.74).
Context episodes are rolled out by PPO checkpoints (ckpt_926 = 30M steps, ~0.25; ckpt_1226 = 40M, ~0.44;
ckpt_1826 = 60M, ~0.60) acting on the true privileged obs. Code: `icl_episodic.py --ctx-policies a:b:c`.
Runs: `logs/runs/icl-peg-*` (jobs 42394316-18, 42394226, 42399580-83).

## Success_once at h=0..4 (last iteration)

| run | h0 | h1 | h2 | h3 | h4 |
|---|---|---|---|---|---|
| full-info, own episodes, success fb | .75 | .75 | .74 | .71 | .73 |
| full-info, ckpts 926/1826/1226/expert, success | .76 | .75 | .74 | .73 | .75 |
| full-info, random ckpt order, success | .74 | .71 | .74 | .76 | .74 |
| full-info, ckpts, final_error | .71 | .74 | .75 | .73 | .74 |
| hidden hole, own episodes, final_error (baseline) | .04 | .72 | .68 | .71 | .67 |
| hidden hole, own episodes, success (baseline) | .06 | .03 | .03 | .03 | .03 |
| hidden, ep0 = ckpt_1226, final_error | .08 | .70 | .70 | .75 | .71 |
| hidden, ep0 = ckpt_926, final_error | .07 | .69 | .71 | .71 | .74 |
| hidden, ep0 = ckpt_1226, success | .05 | .69 | .68 | .71 | .71 |
| hidden, ep0 = ckpt_926, success | .07 | .65 | .67 | .68 | .68 |

## Takeaways
- Full info: student ignores context (flat ~0.74 at all h, equal to expert); suboptimal first students change nothing. Controls (shuf_y/no_y/no_ctx) also match, so there is nothing to search for.
- Hidden hole + checkpoint episode 0: the success-flag runs go from chance (0.03) to 0.65-0.69 at h=1. The checkpoint's trajectory (acting on the true hole) leaks the hole location, so even 1-bit feedback suffices. Student reaches ~expert level.
- Caveat: this is therefore not a feedback-driven search; the context is informative through the demo trajectory. Controls (shuf_y/no_y) were reported by the script but not analysed here; ctx-ablated (no_ctx) is the check that matters.
- Weaker ckpt_926 is slightly worse than ckpt_1226 for success fb (0.65 vs 0.69).

Plot: `logs/renders/icl-peg/peg_ctx.png` (`examples/baselines/ppo/plot_peg_ctx.py`).
Controls (hidden, ep0 = ckpt, success flag): shuf_y/no_y match normal (~0.70 at h>=1); no_ctx falls to ~0.06, so the ckpt trajectory carries the hole. With final_error, shuf_y/no_y fall to ~0.
