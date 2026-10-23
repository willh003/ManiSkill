# Handoff: PushT hidden-goal ICL (2026-10-23)

State: the expert (`logs/runs/pr-full/final_ckpt.pt`) and the transformer ICL student (`logs/runs/icl-terminate/`) work;
details in `experiments/2026-10-23-pusht-hidden-goal-icl.md`.

To reproduce: `sbatch examples/baselines/ppo/slurm/expert_goal_ppo.sbatch` (expert, ~45 min) then
`sbatch examples/baselines/ppo/slurm/icl_episodic.sbatch` (student, ~20 min). Both write to `logs/`.

Gotchas: do not train PPO with `--no-partial_reset`; evaluate with termination (`ppo.py` default now); the goal-range env vars
must match between expert training and the student run (`PUSHT_GOAL_*`).

Next: weaker feedback forms (see `ideas/`), then other POMDP variants. `distill_*.py` and `distill_findings.md` are an earlier
(noisy-observation) line of work; their runs are kept in `logs/runs/pusht-distill-*`.
