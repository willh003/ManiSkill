# ICL student: feedback type comparison on PushTGoal-v1 (2026-10-23)

Question: the `icl-terminate` student (feedback = true final T pose error) recovers the hidden goal after one episode, so extra
episodes do not matter. Does weaker feedback (scalar IoU, expert-action hints) still work, and do extra episodes then help?

Setup: `icl_episodic.py --feedback {iou,hints}`, same as `icl-terminate` otherwise (512 tasks/iter, k=4, expert `pr-full`),
30 DAgger iterations. Slurm: `examples/baselines/ppo/slurm/icl_feedback.sbatch` (`FB=iou|hints`), jobs 42218681 (iou), 42218717 (hints).
- `iou`: one scalar, overlap of the final T with the goal (0..1).
- `hints`: no outcome feedback; instead 5 (state, expert action) pairs per episode, at the steps where the student's action
  differed most from the expert's.

Runs: `logs/runs/icl-fb-iou/`, `logs/runs/icl-fb-hints/`, `logs/runs/icl-terminate/` (20 iters, baseline).
Plot: `logs/renders/icl-feedback/feedback_comparison.png` (success_once vs training tasks, panels h=0, h=1, h=4).

## Final eval (success_once by number of previous episodes h)
| feedback | h=0 | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|---|
| true pose error (`icl-terminate`, it 19) | 0.03 | 0.93 | 0.96 | 0.95 | 0.95 |
| expert-action hints (`icl-fb-hints`, it 29) | 0.03 | 0.91 | 0.95 | 0.93 | 0.95 |
| scalar IoU (`icl-fb-iou`, it 29) | 0.03 | 0.06 | 0.06 | 0.07 | 0.05 |

Controls (shuffled / no feedback / no context) are 0.02-0.06 at every h for both new runs.

## Takeaways
- **Hints work almost as well as exact error**: 0.91 at h=1, 0.95 at h=4, ep_len 98 -> ~42. Slower early (0.31 vs 0.52 at it 1,
  ~0.85 vs ~0.93 at it 15) but converges to the same level. Controls at chance, so the gain comes from the hint content.
- **Extra episodes still barely matter** with hints (h=1 -> h=4: +4 points); the early-training gap between h=1 and h=4 is larger
  (e.g. it 2: 0.40 vs 0.57) but closes with training.
- **Scalar IoU never learns**: flat at 3-7% for all h through 30 iterations, indistinguishable from controls, although loss
  falls (0.028 -> 0.024). A single overlap value is ambiguous about the direction and rotation of the error.
- Not tested: IoU with larger k / longer training / bigger context; intermediate signals (coarse direction, IoU + success).
