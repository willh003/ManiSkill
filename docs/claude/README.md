# docs/claude

Notes kept by and for coding agents. Layout (see repo-root `CLAUDE.md`):

- `experiments/` one file per experiment: what was run, results, pointers to plots/checkpoints (in `logs/`)
- `implementations/` how the code works
- `info/` stable facts (cluster, logging, conventions; `remote-workflow.md` = how to sync/submit/pull from local)
- `ideas/` things to try
- `handoffs/` state-of-the-project notes for other agents

Run outputs (runs, checkpoints, renders, Slurm logs) live in `logs/`, which is gitignored.

## Code cleanup (2026-10-27)
Experiment-specific scripts were removed (still in git history at commit 872e8b4 and earlier; experiment notes referring to them are historical):
the PushT noisy-observation distillation line (`distill_recurrent.py`, `ppo_noisy_obs.py`, `eval_obs_noise.py`, `eval_student.py`, plots, `distill_findings.md`),
PushT sanity/diagnostic scripts, expert video/drift, and the per-experiment plot scripts. Removed from `icl_episodic.py`: `iou`/`final_error_tcp`/`init_error_tcp`
feedback, `--hide-cube`, `--hide-grasp`, `--ctx-features`. Generic replacements: `plot_icl.py`, `plot_ppo_curves.py`, `slurm/icl.sbatch`; task classes live in `icl_tasks.py`.
See `implementations/icl-code-layout.md`.
