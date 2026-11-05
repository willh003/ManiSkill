# Peg hold-out test: ICL (per-step feedback) vs BC (2026-11-04)

Implementation: `implementations/checkerboard-holdout.md` (`icl_checker.py`). Expert: `peg-expert-r1/strong_ckpt.pt` (wide_corner: 0.867 white / 0.865 black, single attempt). All numbers: 512 fixed eval goals per region, mean over 3 seeds.

## 1. Original 2x2 checker (box range x +-0.05, y 0.2..0.4): no gap
Jobs 42837410-15 (`logs/runs/checker-{icl,bc}-s{1,2,3}`). BC is ~0.99 pass@1 on black (expert 1.00): interpolating the hole pose is easy. Not a useful test.

## 2. Harder splits (`--split`, `icl_tasks.SPLITS`), BC only (jobs 42840114/15/20, 20 iters, 1 seed)
Wider box range (x +-0.15, y 0.1..0.5). Expert on test region: wide_ring 0.85, wide_half 0.82, wide_corner 0.88 (wide_yaw 0.01 and wide_all 0.22 dropped: expert fails).
BC black pass@1 / pass@2: wide_ring 0.78 / 0.87, wide_half 0.59 / 0.73, **wide_corner 0.35 / 0.48** (white 0.84 / 0.94). Chosen: wide_corner (train one corner quadrant, test the opposite).

## 3. wide_corner, 3 seeds each, 30 iters (jobs 42843514-900; plots `logs/renders/checker/wide_corner_icl_vs_bc_4panel.png`, `bc_wide_corner_curves_final.png`, `icl_wide_corner_curves.png`)
Last-10-iteration means (black pass@1 / pass@2): BC 0.42 / 0.56 (seeds 0.32/0.44, 0.51/0.66, 0.43/0.59); ICL 0.25 / 0.47 (0.24/0.56, 0.16/0.31, 0.34/0.54). White: both ~0.84 / 0.94.
- ICL retry gain is larger (+0.22 vs +0.14) but its attempt 1 is weaker, so ICL pass@2 does not beat BC. Seed-to-seed spread on black (0.15-0.2) is systematic (each seed settles at its own level), not eval noise; white is identical across seeds. Source (init vs data) not isolated.
- ICL controls (seed 1): no_y / shuf_y remove the attempt-2 gain, so the feedback is used.

## 4. Fine-tuning on black demos (`--finetune-ckpt`, jobs 42847929-42848423, `logs/runs/checker-ft-{icl,bc}-s{1,2,3}`, plot `logs/renders/checker/wide_corner_finetune.png`)
Start from each final checkpoint. n in {10, 50, 200} black goals (nested, first n where the expert succeeds, same for all checkpoints). 1000 steps, lr 1e-4, batch 64, no white replay.
BC: n expert demos. ICL: improvement traces, context = the policy's own attempt 1 (no context, true pose; kept even if it succeeds, 10-44% did) with per-step feedback -> query = expert success demo; each step uses the context w.p. 0.5 and pose dropout 0.5 as in pretraining.
Black pass@1 / pass@2 (n = 0, 10, 50, 200):
- ICL: 0.26/0.49, 0.48/0.67, 0.71/0.82, 0.75/0.85
- BC:  0.43/0.56, 0.62/0.78, 0.77/0.85, 0.81/0.87
White pass@1 (forgetting): ICL 0.84 -> 0.31, 0.39, 0.40; BC 0.84 -> 0.63, 0.55, 0.51.
ICL never beats BC after fine-tuning; both gain a lot on black from few demos and both forget white (ICL more).

Black pass@2 at 200 demos: BC 0.87 ~ expert 0.865 (expert is a single attempt, so the dashed line on pass@2 panels is only a reference). The ICL-BC black gap shrinks 0.17 -> 0.06 pass@2 (n=0 -> 200); ICL's retry gain shrinks (0.23 -> 0.10 at n=200) as attempt 1 improves.
ICL forgot white most at n=10 (pass@1 0.31). Checkpoints with fewer attempt-1 successes on the demo goals forgot more (3 ckpts, not conclusive).

## Conclusion / caveats
- Hypothesis (ICL pass@2 > BC pass@2 on held-out black) not supported in this setup: pretrained, ICL attempt 2 uses feedback (controls confirm) but weaker attempt 1 cancels the gain; after fine-tuning BC >= ICL everywhere.
- Fine-tuning has one config (lr 1e-4, 1000 steps, no white replay); 3 checkpoints only. ICL ft keeps attempt 1 even if it succeeded.
- Pretraining detail: context states are masked together with the query (one dropout draw per sample); feedback y and context actions are never masked. At eval nothing is masked.
- Next candidates: white replay during ft, ICL ft without pose dropout, data-seed vs init-seed ablation, more seeds.
