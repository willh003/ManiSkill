# Offline BC vs ICL, whole workspace (2026-11-04)

Code: `examples/baselines/ppo/icl_offline.py` (reuses `Policy`, `rollout`, `make_ctx` from `icl_checker.py`). Runs: `logs/runs/offline-s{1,2,3}/results.json`. Plot: `logs/renders/offline/bc_vs_icl.png`.

## Setup
- Goals: whole wide workspace (x +-0.15, y 0.1..0.5, normal yaw; `wide_corner` ranges, region "all"). Expert (`peg-expert-r1`) solves 0.896 of the 512 fixed eval goals.
- Per seed: 200 goals where the expert succeeds (nested prefixes N = 10..200). Optimal = expert trajectory on the true goal. Suboptimal = expert acting on a random wrong goal (failures only, resampled until it fails), feedback = per-step (hole - head, head, cos/sin yaw) with the true hole.
- BC: N optimal trajectories, no context. ICL: same N optimal trajectories as targets, context = the paired suboptimal trajectory + feedback w.p. 0.5, empty otherwise. Student always sees the true goal; no masking, no online data.
- 3000 grad steps, batch 64 trajectories, lr 3e-4 cosine, same net/steps for both. 3 seeds, 512 eval goals.
- pass@2: ICL attempt 2 is conditioned on attempt 1 + feedback; BC attempt 2 is an independent reset.

## Results (mean of 3 seeds; pass@1 / pass@2; expert 0.896)
| N | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| BC | 0.000/0.001 | 0.000/0.001 | 0.004/0.007 | 0.019/0.031 | 0.071/0.105 |
| ICL | 0.000/0.000 | 0.000/0.000 | 0.006/0.007 | 0.013/0.017 | 0.073/0.097 |
| ICL no_ctx (pass@2) | | | 0.012 | 0.046 | 0.134 |

## Takeaways
- Neither method generalises at these sizes: the goal is 3-D (x, y, yaw) and 200 demos reach only ~7% pass@1 (loss ~0 = memorisation of the demo goals). The comparison is dominated by this regime, so it says little about ICL's retry benefit.
- ICL pass@1 ~ BC pass@1 (the context-dropout mixture does not hurt or help attempt 1). ICL pass@2 <= BC pass@2: the failure context does not help; replacing it with no context (`no_ctx`) gives a higher pass@2 than using it, i.e. the context slightly misleads.
- Why: the student already sees the true goal, and the context (a wrong-goal trajectory) is paired with a different target distribution only through the goal, so nothing forces it to be used. Pass@2 gains for BC/no_ctx come from reset noise on a deterministic policy.
- Next: more steps / larger N (1k-10k) or a narrower workspace so BC is non-trivial; check whether context helps once pass@1 is ~0.3-0.6; consider making the correction informative (e.g. optimal action residuals) or hiding the goal at test time.
