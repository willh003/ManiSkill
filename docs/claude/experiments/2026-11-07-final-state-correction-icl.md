# Final-state-correction ICL with MLPs (2026-11-07, seed 1)
Script `examples/baselines/ppo/icl_final_state.py` (plot `icl_final_state_plot.py`), run `logs/runs/icl-final-state` (results.json, ckpts `icl-N*.pt`, `bc-visible.pt`, `rollouts.pt`), log `logs/icl-final-state.log`, plot `logs/runs/icl-final-state/plots/success_vs_N.png` (copy in `logs/renders/final-state/`).

Setup: 500 expert demos (uniform configs) -> visible-goal MLP BC (3x512). BC rolled out from N fresh uniform configs; keep each final state (hole masked). Each rollout is paired with a RANDOM expert demo, resampled every batch (`--resample`, default): correction = expert final peg pose (7-d) - BC final peg pose; MLP input = [masked state, has_ctx, norm(final state 43 + correction 7)], target = expert action at every step of the paired demo (goal hidden; recoverable only as final peg + correction). Context dropped w.p. 0.5 in training; no BC regulariser. Eval (512 fresh configs): attempt 1 hidden-goal, no ctx; attempt 2 ctx = attempt-1 final state + (expert final peg on the same config - attempt-1 final peg).

Eval references: expert 0.896, visible-goal BC (500 demos) 0.459 (it solves 0.465 of its own 30000 uniform rollouts).

| N | pass@1 | pass@2 | attempt 2 | no ctx / shuffled pass@2 | train loss |
|---|---|---|---|---|---|
| 100 | 0.012 | 0.047 | 0.035 | 0.014 / 0.014 | 0.00535 |
| 300 | 0.004 | 0.037 | 0.033 | 0.006 / 0.018 | 0.00534 |
| 1000 | 0.006 | 0.041 | 0.035 | 0.006 / 0.010 | 0.00534 |
| 3000 | 0.002 | 0.039 | 0.037 | 0.002 / 0.014 | 0.00458 |
| 10000 | 0.010 | 0.037 | 0.029 | 0.018 / 0.021 | 0.00523 |
| 30000 | 0.008 | 0.025 | 0.018 | 0.012 / 0.016 | 0.00607 |

Flat ~0.04, barely above controls, far below BC 0.46: the MLP does not learn to use final peg + correction as the goal (train loss ~0.005 stays above the visible BC's 0.0016). Not diagnosed yet (ideas: feed final_peg + correction explicitly; drop the 43-d final state from the context; oracle single-token check on this MLP).

## Follow-up: explicit final pose + correction, no final state, context dropout fixed (2026-11-07)
New `--ctx-kind` options in `icl_final_state.py`: `pose` = context [attempt-1 final peg pose (7), correction = goal peg pose - final peg pose (7)] (raw final state removed); `goal` = [expert final peg pose] only; `hole` = [exact hole pose] only (references with the same MLP/training). Runs: `logs/runs/icl-final-{pose,goal,hole}[-nodrop]`, logs `logs/icl-final-*.log`, plot `logs/runs/icl-final-pose-nodrop/plots/success_vs_N.png`.

**Context dropout (p=0.5) broke everything**: with it, even `hole` (exact hole pose as context) got 0.031 pass@2 (train loss 0.0045); with `--ctx-drop 0` it gets 0.484 (loss 0.0017; visible-goal BC 0.459). Cause not understood (guess: unsolvable no-context samples dominate the clipped gradient). Earlier flat results (state-ctx run, pose and goal with dropout) are therefore not informative about the idea. With ctx-drop 0 attempt 1 sees an empty (untrained) context; pass@1 ~0.005 regardless.

References (ctx-drop 0): exact hole pose only 0.484; goal peg pose only (expert final peg pose) 0.346; visible-goal BC 0.459; expert 0.896.

Pose-context student (ctx-drop 0), pass@2 / attempt 2 / no-ctx / shuffled:
| N | pass@2 | attempt 2 | no ctx / shuf | train loss |
|---|---|---|---|---|
| 100 | 0.420 | 0.414 | 0.008 / 0.008 | 0.00219 |
| 300 | 0.402 | 0.400 | 0.006 / 0.004 | 0.00213 |
| 1000 | 0.395 | 0.391 | 0.004 / 0.006 | 0.00219 |
| 3000 | 0.316 | 0.316 | 0.004 / 0.006 | 0.00234 |
| 10000 | 0.244 | 0.238 | 0.010 / 0.020 | 0.00298 |
| 30000 | 0.135 | 0.129 | 0.008 / 0.016 | 0.00310 |

Works at small N (0.42, above the goal-peg-pose-only reference 0.35) but gets WORSE with more rollouts, and train loss rises with N. Hypothesis: with few distinct final poses the MLP memorises them and uses the correction alone; with many it must learn the exact sum final+correction within 6000 steps and underfits (not tested). Next: add the explicit sum (goal pose) as an input, or train longer for large N.

## Train/val curves (pose context, ctx-drop 0; 2026-11-07)
Run `logs/runs/icl-final-pose-curves` (curves.json, plot `plots/curves.png`, copy `logs/renders/final-state/curves.png`; script `icl_final_state_curves.py`). Val set = 200 held-out expert demos (never trained on); ICL val pairs each with held-out BC rollouts (last 2000 of the rollout pool, 1000 fixed pairs).
- Final train / val action MSE: BC 0.0017 / 0.0099; ICL N=100 0.0030 / 0.0118; N=1000 0.0029 / 0.0099; N=30000 0.0034 / 0.0110.
- Train loss still falling at step 6000 (partial underfit), but val loss plateaus by ~2000-3000 steps at ~0.010-0.012 for BC and all ICL runs: train/val gap ~4-6x = overfitting to the 500 expert demos. Longer training would only lower train loss.
- Val loss does NOT worsen with N (N=30000 ~ N=1000), so the "success falls with N because 6000 steps underfit the sum" hypothesis is not supported by val loss. Grad-norm spike at N=30000 around step 1300.
- Success in this rerun: pass@2 0.182 (N=100), 0.260 (N=1000), 0.168 (N=30000) vs 0.420 / 0.395 / 0.135 in the earlier run of the same config (only difference: extra held-out data collection changed the global RNG stream). Run-to-run noise is huge, so the earlier monotone decline with N is not reliable; need several seeds per N.

## 12000 steps (2x), pose context, ctx-drop 0 (2026-11-08)
Run `logs/runs/icl-final-pose-12k` (log `logs/icl-final-pose-12k.log`, curves `plots/curves.png`; `--steps 12000` also applies to the BC baseline). Eval: expert 0.900, visible-goal BC 0.549 (0.459 at 6000 steps).
| N | pass@2 | attempt 2 | no ctx / shuf | final train / val MSE |
|---|---|---|---|---|
| 100 | 0.502 | 0.500 | 0.004 / 0.006 | 0.0008 / 0.0093 |
| 1000 | 0.324 | 0.320 | 0.008 / 0.014 | 0.0009 / 0.0101 |
| 30000 | 0.223 | 0.221 | 0.004 / 0.008 | 0.0010 / 0.0106 |
BC final train / val 0.0006 / 0.0099. Train loss drops ~3x vs 6000 steps (0.003 -> 0.0009) but val loss is unchanged (min val reached ~step 6000 and flat after): the extra steps only widen the train/val gap (10x). Success across the three pose-context runs (pass@2, N=100/1000/30000): 6000 steps run A 0.42/0.40/0.14, run B 0.18/0.26/0.17, 12000 steps 0.50/0.32/0.22: N=100 ranges 0.18-0.50 between runs, so differences < ~0.15 are noise; the 12k improvement is not established, nor is the decline with N (though all three runs have N=100 >= N=30000 on average).

## 20k-step N=1000 run with eval every 1k steps (2026-11-09)
Question: is the "worse with more steps / N" pattern undertraining? Run: `icl_final_state.py --ctx-kind pose --ctx-drop 0 --ns 1000 --steps 20000 --eval-every 1000 --out-dir logs/runs/icl-final-pose-20k` (cosine LR over 20k, so mid-run evals are at un-annealed LR). Trace: `trace.json`; plots: `plots/trace.png` (success vs step), `plots/curves.png` (train/val loss). Log `logs/icl-final-pose-20k.log`.
- Visible-goal BC (success every 1k): 0.22@1k, 0.36@2k, 0.41@3k, 0.45@5k, 0.52@7k, 0.555@8k, then flat 0.54-0.58 to 20k. Final eval 0.566; expert 0.895.
- ICL N=1000 pass@2: 0.11@1k, 0.26@2k, 0.40@4k, 0.44@7-10k, 0.47-0.49@12-15k, 0.50-0.52@16-19k, 0.47@20k. Final controlled eval: pass@2 0.498 (attempt2 0.496), no_ctx 0.006, shuf 0.004. Val MSE bottoms at 0.0091 (~step 8k) and drifts to 0.0095; train MSE 0.00035.
- Reading: earlier 6k-step runs WERE undertrained for success (N=1000 was 0.32 at 12k in the previous run, 0.40 at 6k here; success keeps rising to ~0.5 while val MSE is flat/slightly worse). At 20k N=1000 matches the visible BC (0.50 vs 0.57), i.e. no ICL-specific gap remains. Single-eval noise is ~±0.03-0.05 here.

## 3 seeds x N={100,1000,30000} at 20k steps (2026-11-09)
`icl_final_state.py --ctx-kind pose --ctx-drop 0 --ns 100,1000,30000 --steps 20000 --eval-every 5000 --seed {1,2,3}`; runs `logs/runs/icl-final-pose-20k-s{1,2,3}`, logs `logs/icl-final-pose-20k-s*.log`, aggregate plot `logs/renders/final-state/seeds-20k.png` (`icl_final_state_seeds.py`). Seeds change the 500 demos, BC, rollouts and init; the 512 eval configs are fixed.
Final pass@2 (per seed; mean, sd): visible BC .562/.549/.529 (.547); N=100 .535/.490/.469 (.498, .034); N=1000 .455/.363/.449 (.423, .051); N=30000 .301/.328/.307 (.312, .014). Controls (no_ctx, shuf) <= 0.02.
Convergence (mean pass@2 at 5k/10k/15k/20k): N=100 .38/.45/.48/.50; N=1000 .32/.39/.41/.42; N=30000 .23/.28/.29/.31. Still creeping up slightly at 20k, gains over the last 5k are ~0.01-0.02 (<noise).
Reading: the decline with N is real and consistent across seeds (N=30000 is below N=100 in every seed, ~0.19 gap vs sd ~0.03), not undertraining. The 6k-step undertraining explained the gap to the BC, not the trend with N.
