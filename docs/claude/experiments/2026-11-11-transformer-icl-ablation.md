# Transformer offline-ICL ablation (2026-11-11, 3 seeds)
Script `examples/baselines/ppo/icl_transformer.py` (plot/table `icl_transformer_plot.py`; sbatch `slurm/icl_tf.sbatch`, `icl_mlp.sbatch`). Runs `logs/runs/icl-tf-{fcfc,fcfc_hist,hist_c,hist_c_noisy}-s{1,2,3}` (results.json, curves.json, trace.json, ckpts `*.pt`; seed 1 local, seeds 2-3 klone, `fcfc` runs also hold transformer BC `tf-bc.pt`), MLP N=10000 refs `logs/runs/icl-norm-pose_sum-N10000-s{1,2,3}`. Plot `logs/renders/icl-tf/summary.png`, table `logs/renders/icl-tf/summary.md`. Tracker artifact experiment `icl-tf-ablation`.

## Setup
Same data pipeline as `icl_final_state.py` (500 expert demos -> MLP BC rollout source -> N suboptimal rollouts, each paired with a random expert demo, resampled per batch; context dropout 0; expert-stat normalisation, clip +-5). Transformer = encoder over [obs token, context pose tokens] (7-d peg poses, learned per-slot embeddings), action read from the obs token. f = final peg pose of the failed attempt, g = expert final peg pose on that config, c = g - f, hist = peg pose at steps 0,10,..90.
Kinds: **T1** [f,c,f+c]; **A** [f,c,f+c]+hist(10); **B** [hist(10), f, c]; **C** = B with c*s, s~U(.9,1.1) redrawn per sample (train) / per attempt (eval). Eval: recursive attempts k=1..8, attempt k sees only attempt k-1 (sliding window of 1), fresh s per attempt, 512 eval configs; distance = final peg xyz to g xyz (m); "best" = min over attempts <= k.
Final config: d=128, 2 layers, 4 heads, 10k steps, AdamW lr 3e-4 cosine, warmup 500, bf16 autocast, batch 128 demos x 100 steps.

## Sanity gate (transformer >= MLP): passed, but only after shrinking the model
First attempt (d=256, 4 layers, 20k steps): transformer BC .66 (>MLP .55) but T1 N=1000 pass@2 only .254 (MLP .52); train loss 1e-5, val .0117. Not precision (fp32 .29), not fixed by dropout .1 (.29) or weight decay .1 (.32). Oracle single-token contexts show memorisation: exact hole pose token peaks .537 at 5k steps then decays to .42; expert-final-peg "goal" token .33. Sweep at N=1000 (pass@2): 6k steps .39, lr1e-4 .41, dropout+wd .35, **d=128 L=2 10k steps .514** -> adopted for all runs. Chosen on seed 1 N=1000 only (same seed/N as reported cell, mild selection bias).

## Results (mean +- sd over 3 seeds)
Refs: TF BC (visible goal, 500 demos) .824/.838/.836; MLP BC .49-.54 (rollout source); MLP [f,c,f+c] 20k steps pass@2 .523 / .523 / .507 at N=100/1000/10000.
| method | N | pass@2 | pass@4 | pass@8 | best dist @2 / @8 (m) |
|---|---|---|---|---|---|
| T1 [f,c,f+c] | 100 | .490+-.018 | .626 | .681 | .049 / .028 |
| T1 | 1000 | .504+-.044 | .656 | .712 | .050 / .026 |
| T1 | 10000 | .501+-.019 | .644 | .706 | .049 / .026 |
| A +hist | 100 | .484+-.070 | .604 | .658 | .055 / .032 |
| A | 1000 | .432+-.030 | .561 | .624 | .061 / .036 |
| A | 10000 | .443+-.058 | .583 | .641 | .061 / .033 |
| B hist+c | 100 | .044+-.011 | .148 | .227 | .120 / .076 |
| B | 1000 | .026+-.014 | .061 | .098 | .154 / .105 |
| B | 10000 | .027+-.003 | .074 | .129 | .139 / .090 |
| C noisy c | 100 | .036+-.006 | .109 | .201 | .124 / .077 |
| C | 1000 | .022+-.005 | .055 | .093 | .160 / .110 |
| C | 10000 | .023+-.009 | .084 | .123 | .143 / .092 |
no-context / shuffled-context pass@2 <= .016 everywhere (context is used). Full table with angles: `logs/renders/icl-tf/summary.md`.

## Reading
- **Transformer vs MLP**: T1 matches the MLP (.49-.50 vs .51-.52, within seed sd) and is flat in N for both; transformer BC is much stronger than MLP BC (.83 vs ~.5). The gate is "at least as good": equal for ICL (not clearly better), better for BC. The big transformer failing the gate was overfitting to the 500 demos, not a bug (oracle checks above).
- **A (add history)**: slightly lower than T1 at N>=1000 (.43-.44 vs .50, ~1.5 sd), no clear N trend; hypothesis "doesn't change anything" roughly holds, with a possible small loss (more tokens to memorise).
- **B (drop f+c)**: hypothesis NOT supported: pass@2 collapses to .03-.04. A control [f,c] with no history (`icl-tf-fc-s1`, seed 1 only, N=100/1000) gets .27/.23, so removing f+c costs about half, and adding the history tokens on top removes the rest. Whatever the cause (computing the sum across tokens with many distractor tokens; memorisation), the goal is not reliably recovered from final frame + vector.
- **C (noisy c, sliding window, pass@8)**: same as B (.02-.04 pass@2, .09-.20 pass@8). Best-so-far distance does fall with k (N=100: .124 -> .077 m) but the per-attempt distance is flat from k=2 on (~.15-.2 m), so the decrease is best-of-k over more samples, not iterative refinement. Noise itself costs little (C vs B within noise).
## Caveats
Student's later attempts are off-distribution (trained only on BC failures). Model size/steps were tuned on T1 seed 1 N=1000 and reused for B/C unchanged; B/C might need different capacity/regularisation (not explored). Seeds change demos, BC and init; eval configs fixed. MLP BC rollout source is retrained per run (seed-matched, not bit-identical across kinds).
