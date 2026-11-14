# Debug plan: why is T1 ICL pass@2 (~.50) far below transformer BC (.83)?

**Why it is surprising.** pass@2 for our ICL models is the success of attempt 2, i.e. the policy conditioned on a context that (for T1) contains g = f+c, the expert's final peg pose, in a fresh episode with the same goal as the BC eval. Transformer BC sees the hole pose directly in its obs token and gets .83. Caveats: g is the expert *final peg pose* (a deterministic-ish function of the hole pose, not the hole pose itself); and BC's val loss is also 6x its train loss, so memorisation is not ICL-specific. But an earlier oracle (exact hole-pose token, big model) reached only .537, so the gap already exists with the *simplest possible* context. Hypothesis: the policy learns badly from a context token (vs. from an obs feature), independent of what the context contains.

**Metric for every cell:** pass of the context-conditioned attempt (= pass@2), 3 seeds, N=1000 where N matters. Reference line = TF BC .83. Also log train/val loss. Target for a "working" ICL step: within ~.05 of BC.

## Ladder (each rung changes one thing; stop at the first rung that drops)
- **R0 sanity (architecture):** BC with the hole pose supplied *through the context token* instead of the obs token, no rollouts involved (kind `hole`). If this is << .83 the problem is the token pathway/normalisation/optimisation, not the ICL data.
- **R1 `[g]`:** one token = expert final peg pose (kind `goal`; identical info to `[f+c]`). Compare to R0 to see how much is lost going hole -> g (g is a noisy function of the hole).
- **R2 `[f, g]`:** add f as a distractor (does irrelevant context hurt?). Also `[f_random, g]` to separate "extra token" from "extra *correlated* token".
- **R3 `[f, c]` (kind `fc`, existing .27/.23):** the model must compute g = f+c across tokens.
- **R4 `[f, c, f+c]` = T1 (.50):** token ordering variants (g first / last) and per-slot embedding on/off.
- **R5 +history (A), history+c (B).**

## Orthogonal checks to run at the first rung that drops
1. Training-data pairing: context built from a rollout paired with a *random* expert demo whose g is independent of the rollout. Check train-loss/val-loss on pairs and the fraction of val loss explained by the context (loss with correct vs. shuffled ctx).
2. Normalisation: ctx stats from the expert demos with clip +/-5; quaternion sign (q and -q) and the hole-vs-peg pose frames.
3. Optimisation: steps (memorisation peaks early: the exact hole token peaked at 5k steps then decayed), lr, weight decay, d, dropout; eval pass@2 along training (trace) to see if the optimum is early.
4. Memorisation: 500 demos only; try 2000 expert demos for R0/R1 (is the gap data-limited?). Compare BC with 500 vs 2000 demos too.
5. Where the context enters: obs-token concat vs cross-attention vs token; read action from obs token vs mean pool.

## Order of work
R0, R1 and the existing `fc` first (cheap, kinds already implemented: `hole`, `goal`, `fc`), at the adopted d=128/L=2/10k steps, 3 seeds. Then fill in R2 variants and the orthogonal checks only where a drop appears.

## Results R0/R1/R3 (d=128, L=2, 10k steps, N=1000, 3 seeds; `scripts/run_icl_debug_r0.sh`, logs/runs/icl-tf-dbg-r0-s{1,2,3})
| rung | context | pass@2 per seed | mean | pass@8 | val loss |
|---|---|---|---|---|---|
| ref | TF BC, hole in obs | .824/.838/.836 | .833 | - | .0087 |
| R0 | `[hole pose]` | .717/.686/.703 | .702 | .744 | .0110 |
| R1 | `[g]` (= `[f+c]`) | .490/.531/.527 | .516 | .571 | .0139 |
| R3 | `[f, c]` | .244/.195/.244 | .228 | .550 | .0121 |
| T1 | `[f, c, f+c]` | (main grid, N=1000) | .504 | .712 | |
Reading: the ICL pathway already loses ~.13 vs BC with the exact hole pose as the only token (R0), loses another ~.19 going hole -> g (R1), and R1 ~= T1, so f and c add nothing. R3 shows computing f+c across tokens is learned poorly (pass@2 .23, although pass@8 .55 means it gets there with retries via attempt noise). Note g reaches the same hole-peg distance (.05) as hole (.04) at best-of-k, so g is informative; the issue is how reliably the policy uses it in one attempt. Next: orthogonal checks at R0 (normalisation, steps, 2000 demos, context pathway).

## Interim (seed 1 only, N=1000; seeds 2-3 still running via scripts/run_icl_debug_batch.sh, logs/icl-tf-dbg-{perm,cat,long,demos}-s*.log)
- **Context pathway is the main culprit.** `--cat` (context concatenated to the obs token instead of separate context tokens): hole .826 (= BC .83), goal .684, T1 [f,c,f+c] .721, vs. separate tokens .70 / .52 / .50. 
- Longer training (30k steps) hurts: hole .650, goal .441 (memorisation).
- Permutations with separate tokens, seed 1: [f,g] .426, [rand,g] .449, [g,f] .426 (extra token costs ~.09 vs [g] .52; correlation with g irrelevant; order irrelevant).
Reading: the transformer encoder over [obs, ctx tokens] is a worse way to inject context than feeding it into the obs embedding; likely the obs token has to learn attention-based retrieval from few demos. 2000-demo runs and the remaining permutations pending.

## Final results, batch 2 (3 seeds, N=1000, pass@2; `scripts/run_icl_debug_batch.sh`, logs/runs/icl-tf-dbg-{perm,cat,long,demos}-s*)
| setup | pass@2 | pass@8 |
|---|---|---|
| TF BC hole visible (500 demos) | .833 | - |
| **tokens** `[hole]` / `[g]` (R0/R1) | .702 / .516 | .744 / .571 |
| tokens `[f,g]` / `[g,f]` / `[rand,g]` / `[c,g]` | .447 / .429 / .465 / .454 | .64 / .62 / .62 / .64 |
| tokens `[g,f,c]` / `[c,f,g]` / T1 `[f,c,f+c]` | .514 / .511 / .504 | .71 / .71 / .71 |
| tokens `[f,c]` (R3) | .228 | .550 |
| **concat** `[hole]` / `[g]` / T1 | **.824** / .676 / .734 | .850 / .712 / .838 |
| tokens, 30k steps `[hole]` / `[g]` | .664 / .436 | .718 / .508 |
| tokens, **2000 demos** `[hole]` / `[g]` (BC 2000 demos: .897 TF, .839 MLP) | .880 / .758 | .898 / .784 |
Findings: (1) the context-as-token pathway is the largest loss: exact hole pose via concat = BC (.824 vs .833), via tokens .702; (2) hole -> g costs ~.15-.19 in either pathway (g is the expert's final peg pose, not the hole); (3) extra tokens cost ~.06-.09 regardless of correlation or order; computing f+c from separate tokens is much harder ([f,c] .23) but [g,f,c]/[c,f,g] (g present) ~= T1; (4) longer training hurts (memorisation); (5) more expert demos help both (2000 demos: hole token .88).
Implication: use `--cat` (or otherwise inject context at the obs embedding) for the ICL ablations, and more expert demos; T1/A/B/C numbers in the main grid are depressed by the token pathway. Not yet re-run: A/B/C and the MLP comparison under `--cat`.

## Final results, batch 3 (3 seeds, pass@2 / pass@8; runs icl-tf-dbg-{catabc,catdemos,slot,cnorm}-s*)
**Variants with the concat pathway (`--cat`, 500 demos)** vs tokens (main grid):
| kind | N=100 | N=1000 | N=10000 | tokens N=1000 |
|---|---|---|---|---|
| T1 [f,c,f+c] | .688 / .82 | .738 / .84 | .697 / .83 | .504 / .71 |
| A +history | .553 / .85 | .683 / .84 | .695 / .85 | .43 |
| B history + c | .458 / .83 | .606 / .84 | .626 / .83 | .03 |
| C B + noisy c | .163 / .60 | .266 / .65 | .282 / .66 | .03 |
MLP [f,c,f+c] pass@2: .523/.523/.507; TF BC (hole visible) .833 pass@2-equivalent. Under concat the transformer T1 beats the MLP (.74 vs .52), A is close to T1 (hypothesis roughly holds at N>=1000), B drops ~.13 below T1 (partly supported: much better than tokens .03, but not "as good"), and C's scale noise now hurts a lot (.27) because the model can use c precisely. pass@8 of T1/A/B ~.83-.85, i.e. at the BC ceiling.
**Concat, 2000 demos (N=1000):** hole .881, goal .799, T1 .805 (BC 2000 demos .897).
**Tokens, controls (N=1000, 500 demos):** no slot embeddings: hole .701, goal .503, T1 .340 (slot embeddings only matter for the 3-token layouts: T1 .504 -> .340); unnormalised context: hole .628, goal .523, T1 .480 (normalisation helps slightly for hole, not a cause of the gap).
Ladder status: R0-R5, token order, slot embeddings, normalisation, training length, demo count and context pathway all run. Conclusion: ICL with separate context tokens was the bug-like bottleneck; concat closes the hole-pose gap completely and recovers most of T1; the rest is hole->g information (.88 -> .80 with 2000 demos) and data (500 demos).

## Replotted with concat pathway (500 demos; `icl_requested_plots.py cat`)
logs/renders/icl-tf/plot{1_t1_vs_bc,2_pass2_vs_N,3_variants}_cat.png (token-pathway originals without suffix). T1 pass@k reaches .82-.84 by k=8 (BC .85) with ~.70 at k=2; transformer T1 > MLP at every N (.69-.74 vs .51-.52) and neither depends on N; A, B, T1 overlap by k=4; C (noisy c) stays low (.60-.66 at k=8). The MLP concatenates its inputs by construction, so it is already at its "best pathway". 2000-demo full grid (A/B/C, MLP) not run.

## 2000-demo concat grid (3 seeds; runs icl-tf-dbg-catabc2k-s*, icl-tf-dbg-mlp2k-s*, logs/icl-bc-passk-2k.json; plots logs/renders/icl-tf/plot*_cat2k.png)
TF BC (hole visible, 2000 demos) .886 pass@1 -> .90 at k=8; MLP BC .84. Rollout source is the 2000-demo MLP BC.
| kind | N=100 p@2 / p@8 | N=1000 | N=10000 |
|---|---|---|---|
| T1 [f,c,f+c] | .785 / .88 | .810 / .88 | .811 / .88 |
| A +history | .563 / .88 | .696 / .88 | .751 / .88 |
| B history + c | .528 / .87 | .624 / .88 | .660 / .87 |
| C B + noisy c | .246 / .74 | .316 / .80 | .414 / .82 |
| MLP [f,c,f+c] (2000 demos, 20k steps) | .788 | .790 | .784 |
Findings: with 2000 demos and concat, T1 reaches .81 pass@2 and .88 pass@8 (BC .89/.90), i.e. ~.08 below BC at k=2 and ~.02 at k=8. The MLP now matches the transformer (.78-.79 vs .79-.81), so the transformer is only better than the MLP when data is scarce (500 demos: .74 vs .52); the "transformer >= MLP" sanity check passes. More suboptimal data (N) still barely matters for T1/MLP, but A/B/C improve with N (.56 -> .75 for A, .25 -> .41 for C). A and B lag T1 at k=2 and catch up by k=5.

## Variant C with biased scale U(1.1,1.3) (concat, 2000 demos, 3 seeds; runs icl-tf-dbg-cbias2k-s*; plot logs/renders/icl-tf/plot3_variants_cat2kb.png)
pass@2 / pass@4 / pass@8 (C = U(.9,1.1) in parentheses for pass@2/8): N=100 .307 / .607 / .777 (.246/.74); N=1000 .443 / .763 / .831 (.316/.80); N=10000 .458 / .742 / .809 (.414/.82). Best-so-far peg-goal distance 0.14-0.15 m at k=1 -> .05-.07 at k=2 -> ~.017 at k=8. Biased (over-shooting) corrections are *not* worse than unbiased noisy ones; they are better at small k (the model can learn to compensate a systematic overshoot of ~20%, whereas zero-mean noise cannot be undone), and the two converge by N=10000 / k=8. Both stay below T1/A/B at k<=4.

## 100 expert demos (concat pathway, 3 seeds, klone a40; runs icl-tf-dbg-c100-{fcfc,fcfc_hist,hist_c,hist_c_noisy,cbias}-s*, icl-tf-dbg-mlp100-s*, logs/runs/icl-bc-passk-100.json; plots logs/renders/icl-tf/plot{1,2,3}_*_cat100.png)
BC, hole visible: TF BC .123/.164/.184 (final), pass@1/2/8 .141/.160/.173 (flat; hole hidden = 0); MLP BC .05 (the rollout source); expert .90. pass@2 / pass@4 / pass@8:
| kind | N=100 | N=1000 | N=10000 |
|---|---|---|---|
| T1 | .176 / .264 / .331 | .154 / .237 / .288 | .180 / .269 / .319 |
| A +history | .172 / .311 / .411 | .178 / .273 / .346 | .189 / .296 / .374 |
| B history + c | .055 / .148 / .242 | .076 / .150 / .225 | .052 / .103 / .157 |
| C s~U(.9,1.1) | .022 / .044 / .083 | .018 / .048 / .089 | .011 / .036 / .068 |
| C-biased U(1.1,1.3) | .023 / .057 / .098 | .034 / .071 / .110 | .029 / .061 / .093 |
| MLP pass@2 | .045 | .031 | .047 |
Findings: with only 100 demos everything is weak (BC .14-.17; the policy itself rarely inserts even with the hole visible). T1 pass@2 (.15-.18) ~= BC (.16) and T1 exceeds BC by k=8 (.29-.33 vs .17), i.e. retries with corrections help beyond what the BC policy can do alone; A is the best variant (.35-.41 at k=8, history helps when data is scarce, unlike at 500/2000 demos); B and C are far worse and the biased C is only marginally better than unbiased; the MLP (.03-.05) is below the transformer (.15-.18) and no better than its own BC rollout source. N does not matter. Note the 100-demo BC rollout source is weak (MLP BC .05), so context rollouts are mostly far-from-goal flings.
