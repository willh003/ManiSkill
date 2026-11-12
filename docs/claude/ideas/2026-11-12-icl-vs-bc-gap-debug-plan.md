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
