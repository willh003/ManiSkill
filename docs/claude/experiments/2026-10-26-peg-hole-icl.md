# ICL student on PegInsertionSideBias-v1: biased / hidden hole pose (2026-10-26)

Env, task and feedback definitions: `docs/claude/implementations/peg-hidden-hole-env-and-icl.md`. Expert: `logs/runs/peg-expert-fast/final_ckpt.pt`
(PPO, success_once ~0.71-0.75; see `2026-10-26-peg-expert.md`). Student: `icl_episodic.py --env-id PegInsertionSideBias-v1`, 512 tasks/iter, 30 iterations,
k=4, 100 steps, d_model 128, 2+2 layers; student obs drops the hole radius. Slurm `slurm/icl_peg.sbatch`, runs `logs/runs/icl-peg-<feedback>-{bias,hidden}/`
(jobs 42385683-88). Plot: `logs/renders/icl-peg/peg_icl.png` (`plot_peg_icl.py`).

## Results (success_once at the last iteration, 512 eval tasks; h = previous episodes in context)
Biased observed hole pose (+-2 cm xy, +-0.1 rad yaw). Expert true pose 0.71; expert told no bias 0.26.
| feedback | h0 | h1 | h2 | h3 | h4 | shuf_y h1 | no_y h1 | no_ctx h1 |
|---|---|---|---|---|---|---|---|---|
| true hole pose | 0.64 | 0.75 | 0.72 | 0.70 | 0.72 | 0.03 | 0.02 | 0.67 |
| expert hints | 0.64 | 0.72 | 0.70 | 0.71 | 0.71 | 0.16 | 0.26 | 0.63 |
| success flag | 0.63 | 0.59 | 0.60 | 0.62 | 0.62 | 0.53 | 0.49 | 0.64 |

Hole pose hidden (box x, y, yaw fixed per task). Expert true pose 0.75; expert told the default box pose 0.04.
| feedback | h0 | h1 | h2 | h3 | h4 | shuf_y h1 | no_y h1 | no_ctx h1 |
|---|---|---|---|---|---|---|---|---|
| true hole pose | 0.05 | 0.72 | 0.68 | 0.71 | 0.67 | 0.01 | 0.00 | 0.07 |
| expert hints | 0.08 | 0.65 | 0.68 | 0.69 | 0.65 | 0.01 | 0.01 | 0.07 |
| success flag | 0.06 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.03 | 0.06 |

## Takeaways
- **Hidden hole**: h=0 is chance; one episode of true-pose feedback gives 0.72 (96% of the 0.75 expert), hints 0.65. Controls (shuffled / no feedback / no
  context) are at chance, so the gain is from the feedback. More episodes do not help (the hole does not move, so h=1 already has the full information;
  the student is limited by the expert's ~0.75). Success flag learns nothing (as for the hidden cube), below h=0 because the context is uninformative.
- **Biased obs**: weak test. The student is already at 0.64 at h=0 (expert told no bias: 0.26) because it trained on biased obs with clean-expert labels and is
  fairly robust to +-2 cm; headroom to the 0.71 expert is only ~0.07, which feedback closes (0.72-0.75 at h=1). With the success flag, context *hurts*
  (0.59-0.62 at h>=1 vs 0.64 at h=0; no_ctx at h>=1 is 0.64-0.67): an uninformative past episode is distracting. A larger bias would give a stronger test.
- All numbers are limited by the ~0.73 expert; a stronger expert (resume runs `peg-expert-r1/r2`) would lift the ceiling. Not re-run with it.
