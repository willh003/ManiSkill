# Recurrent PPO upper bound for hidden-hole peg with end-of-episode feedback (2026-11-17)

Plan `ideas/2026-11-17-rl2-peg-upper-bound-plan.md`; code `implementations/rl2-peg.md`. Tracker experiment `rl2-peg-upper-bound`.
Eval = the 512 fixed tasks of the ICL runs (wide_corner 'all', eval_seed 12345), deterministic, same peg / robot start in every episode.

## Scripted references (`meta_peg_oracle.py`, `logs/rl2-peg/oracle-<cfg>.json`)
Success per episode. oracle = ep 1 prior-mean-hole expert, ep k>1 expert on mean_j(f_j + c_j).
| config | expert (true hole) | prior (blind) | oracle |
|---|---|---|---|
| 2 eps, noiseless (e2-n0) | .900 / .898 | .010 / .010 | .012 / .902 |
| 2 eps, noise .5 cm / .025 rad (e2-n05) | | | .012 / .607 |
| 2 eps, noise 1 cm / .05 rad (e2-n1) | | | .012 / .330 |
| 3 eps, noise 1 cm / .05 rad (e3-n1) | .900 / .898 / .904 | | .010 / .332 / .484 (pass@3 .59) |
The clearance is 3 mm, so a 1 cm error in the hole estimate mostly fails for an expert that trusts it.

## Pilot (local, e2-n0, seed 1): `logs/runs/rl2-peg-pilot-s1`
BC (12 DAgger iterations, ~40 min) then 300 PPO iterations (1024 envs, ~25 s/iter on the 5090, ~2.5 h).
- BC init: ep 1 .004, ep 2 .863 (label MSE ep 2 .004, ep 1 .016: ep-1 labels are the prior expert's, so learnable).
- PPO: ep 2 .90 by iteration 30 (= oracle), .955 by 80, **.984 final** (ep 1 .014). PPO beats the expert it was distilled from (.90), since
  it optimises insertion directly (the expert's failures are jams at the hole entrance, see `2026-10-26-peg-expert.md`).
- Controls (ep 2): other task's (f, c) .004; zeroed feedback .002; GRU state reset at the episode boundary .982. The policy reads the explicit
  feedback; the recurrent memory is not needed when (f, c) stay in the observation (expected for noiseless feedback).
- Compare ICL (`2026-11-16-icl-onpolicy-loop.md`, hole hidden, attempt 1 blind): best pass@2 at 2000 demos ~.82-.87; at 100/200 demos far lower.
  The RL upper bound for the same 2-attempt protocol is .98.

## Grid (klone, 3 seeds each, 300 PPO iterations): e2-n0, e2-n05, e2-n1, e3-n1
Submitted with `scripts/cluster/submit_cmd.sh rl2-peg-<cfg>` (3 seeds in parallel per job, all GPU groups, first to start cancels the rest).
Klone's gscratch was very slow on 2026-11-17 evening: jobs on g3118 and g3117 sat >15 min in uninterruptible IO importing torch; the g3118
job was cancelled and resubmitted with `SBATCH_EXTRA=--exclude=g3118`. Local copies `rl2-peg-{e2-n1,e3-n1}-loc-s1` run meanwhile.

## Results
(pending)
