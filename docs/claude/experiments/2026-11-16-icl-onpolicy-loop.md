# On-policy outer loop for hidden-hole peg ICL (2026-11-16)
Script `examples/baselines/ppo/icl_onpolicy.py`, plots `icl_onpolicy_plot.py` -> `logs/renders/icl-onpolicy/`, job `slurm/icl_onpolicy.sbatch`, submit `scripts/cluster/submit_onpolicy.sh`. Runs `logs/runs/icl-op-<kind>-<pairing>-<supervise>-d<demos>-s<seed>/` (results.json, curves.json, state.pt, checkpoints `tf-bc.pt`, `round<r>.pt`). Tracker experiment `icl-onpolicy-loop`.

## Why
The 2026-11-11..15 runs showed the ICL policy ignores the failure feedback: it was trained once on failures of a visible-hole BC, each paired with a random other task's expert demo. This adds an outer loop of on-policy collection.

## Recipe
- E expert demos (100 / 200 / 2000) + 200 held-out val demos; each demo's task (batch seed, env slot, hole pose) is replayable.
- Round 0: visible-hole transformer BC on the demos, one deterministic rollout per demo task -> pool.
- Round r = 1..8: train the hole-hidden ICL transformer from scratch on the pool (concat pathway, d=128, L=2, 10k steps). Sample = (context from a pooled rollout, expert demo target). No context-free samples (see v1 below). Context norm stats from the current pool. Eval on the 512 fixed eval tasks (same as `icl_fair_bc.py`), k=1..8, attempt 1 blind, attempt k sees attempt k-1. Then K=8 sliding-window rollouts on every demo task (and val task) are appended to the pool (successes included).
- Controls at rounds 1 and 8: another task's attempt-1 f with the exact goal (`randf`), attempt-1 context frozen (`fixed`).
- Pairing: `matched` = context from task t with t's demo; `random` = f/h from any pooled rollout, c recomputed for the target demo's goal.
- Context: `goal` [f+c], `fc` [f,c], `hist_c` [h x10, f, c]. `goal` has no f, so pairing is irrelevant; it is the null control for the loop.
- `--supervise dagger`: half of each batch is the student's own states with expert action labels.

## Pilot (local, fc, 100 demos, seed 1; `logs/runs/icl-op-pilot-fc-{m,r}-d100-s1`)
BC solves .77 of its own demo tasks but .117 of eval tasks, so most round-0 contexts are successes (c ~ 0). pass@8: matched .006 (round 1), .029 (round 2); random .020, .025, .012 (rounds 1-3). Val loss matched .124 -> .047 once on-policy data arrives. Grid therefore uses 8 rounds.

## v1 grid (discarded): 1/8 context-free training samples
First grid used p_noctx=.125 so attempt 1 would be trained. At 100 demos this makes the model fit the demos from the state alone and ignore the context: goal-only pass@8 .004 vs .08 with p_noctx=0 (local check, seed 1, round 1, logs `logs/onpolicy/dbg-goal-p{0.0,0.125}.log`). Grid cancelled and rerun with p_noctx=0. v1 runs kept on klone as `logs/runs/icl-opn-*`.

## Results (v2, 3 seeds each, 8 rounds, eval = 512 fixed eval tasks; attempt 1 is blind so pass@1 ~ .005 everywhere)
Plots `logs/renders/icl-onpolicy/<n>_*_{pass_k,succ}.png` (succ = perf@k); results page published as an artifact (see tracker experiment `icl-onpolicy-loop`).
Round 8, pass@8 / perf@8 (mean of 3 seeds):
| demos | [f,c] matched | [f,c] random | [h,f,c] matched | [h,f,c] random | [f+c] | BC+noise .05 (hole visible) |
|---|---|---|---|---|---|---|
| 100 | .152 / .057 | .129 / .057 | .212 / .098 | .288 / .135 | .128 / .092 | .369 / .158 |
| 200 | .506 / .324 | .517 / .324 | .524 / .289 | .643 / .410 | .420 / .374 | .786 / .510 |
| 2000 | .865 / .794 | .867 / .805 | .881 / .796 | .875 / .814 | .834 / .820 | .956 / .893 |
Outer loop, [f,c] matched pass@8 at rounds 1/2/4/8: 100 demos .005/.249/.178/.152; 200 .033/.574/.514/.506; 2000 .678/.863/.868/.865. Random pairing is flat from round 1 (100: .208 -> .129; 200: .564 -> .517; 2000: .870 -> .867).

## Reading
- **The outer loop does not improve the policy after round 2.** Round 1 matched is near 0 because BC solves 77-96% of its own demo tasks, so the matched round-0 pool is mostly successes (c ~ 0) and there is little failure signal; one round of on-policy data fixes that. After round 2 pass@8 is flat (2000, 200) or falls (100: .249 -> .152). Random pairing gets the round-2 level already at round 1.
- **Why it stalls: the demo tasks are memorised.** On its own demo tasks the round-7 model succeeds by attempt 8 at .65 / .78 / .89 (100 / 200 / 2000 demos) vs .057 / .324 / .794 per attempt on eval tasks. Pool success fraction climbs to .51 / .59 / .73 by round 8, so later rounds add mostly successes and failures that don't look like test-time failures.
- **Pairing: task-matched is not better.** Matched ~= random for [f,c]; for [h,f,c] random is better at 100 and 200 demos (.288 vs .212, .643 vs .524), equal at 2000.
- **Feedback use appears only at 2000 demos.** Per-attempt success rises from attempt 2 to 3 with the real previous attempt (.735 -> .787 [f,c] matched; .706 -> .775 [h,f,c] matched; smaller for random), while a frozen attempt-1 context stays flat (.74) and another task's failure stays flat or lower (.70). At 100 and 200 demos perf@k is flat with every context, as before. pass@k still mostly comes from retry variation: frozen context gives the lowest pass@8 everywhere; another task's failure gives equal or higher pass@8 except for [h,f,c] matched.
- **[h,f,c] matched depends on its own task's history.** Swapping in another task's h and f collapses it (perf .03 at 200 demos, .33 at 2000, vs .29 / .80 real). Matched training always pairs the history with the same task and reset, so the model can rely on the history agreeing with the current episode (e.g. the initial peg pose); this is a same-task cue, not evidence of reading the correction.
- **Context type:** at 2000 demos all within .83-.88 pass@8; [f+c] has the best pass@2 (.821) and the lowest pass@8 (flat after attempt 2). At 100 / 200, [h,f,c] random is best.
- **Demo count dominates** (pass@8 ~.15-.29 / .42-.64 / .83-.88).
- **Still below BC with action noise** at every demo count, even at pass@8 (BC gets the hole on attempt 1; ICL attempt 1 is blind). Compared with the offline T1 of the previous session (pass@8 .287 / .660 / .890 at 100 / 200 / 2000), the on-policy loop is no better.

## Checkpoints
`logs/runs/icl-op-<kind>-<pairing>-demo-d<demos>-s<seed>/checkpoints/{tf-bc.pt, round1..8.pt}` (pulled locally); state.pt (pools) stays on klone.

## Pending
DAgger relabel comparison (`--supervise dagger`, [f,c] matched, 100/200/2000, relabels attempts 2..K only) submitted 2026-11-17.

## Re-eval with attempt 1 = BC (2026-11-17)
`icl_onpolicy_bcfirst.py RUN_DIR` → `RUN_DIR/results_bcfirst.json`; plots `RES=bcfirst icl_onpolicy_plot.py` → `logs/renders/icl-onpolicy-bcfirst/`. `[f, c]` task-matched, 3 seeds. Attempt 1 = the run's `tf-bc.pt` (visible hole, deterministic), attempts 2..8 = `round{r}.pt` with a sliding window.

Round 8 perf@k (attempt 1 → 2 → 8): 100 demos .148 → .063 → .069; 200 demos .502 → .291 → .307; 2000 demos .894 → .796 → .788. BC + noise stays at .15 / .51 / .90. perf@8 by round peaks at round 2 (100: .126, 200: .377), 2000 is flat (~.79–.81). pass@8 at round 2: .345 / .733 / .924 vs BC + noise .369 / .786 / .956. Every ICL attempt is below BC's own first attempt and flat across k: the retries are worse independent draws, not corrections.
