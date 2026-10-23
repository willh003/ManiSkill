# Ideas

## Feedback forms for the epistemic-POMDP student (ordered by information)
1. scalar outcome (success / final IoU) 2. final-state error (done) 3. per-step reward/IoU curve
4. dense expert relabel of the student rollout 5. sparse expert hints (m steps) 6. corrected trajectory 7. quantized "language" hints.
With the current setup (student sees its own final T pose) form 2 is solved by h=1; use 1, 3, 5 to make context length matter.

## Other hidden variables
Hidden dynamics / action bias, perception bias (noisy T-pose offset, see `examples/baselines/ppo/distill_findings.md`).

## Training
- SGS (success-rate-guided goal sampling): fixed set of N goals, Beta-kernel sampling on running success rates (t=0.5, kappa=1, T=2,
  H=100 for manipulation) if expert training on wider goal ranges stalls. Not needed so far (pr-full reaches ~1.0).
- Make training reward the post-success state (e.g. terminate after holding the goal for ~10 steps) to remove the hovering incentive.
