# Cluster + logging conventions (klone)

- Never run compute on login nodes; everything goes through `sbatch` (scripts in `examples/baselines/ppo/slurm/`).
- Accounts/partitions for this user: `gpu-l40-weirdlab` (partition `gpu-l40`, preferred), `gpu-l40s-weirdlab`, `gpu-a40-weirdlab`,
  then `ckpt-weirdlab`. Per-account limits bite: a pending job reason `AssocGrpGRES`/`AssocGrpCpuLimit` means the account's
  GPU/CPU cap is in use; check `squeue -u $USER` before queueing a batch of sweeps.
- Slurm stdout: `logs/slurm/%x-%j.out`. Runs: `logs/runs/<name>/`. Renders/figures: `logs/renders/<topic>/`.
  `logs/` is gitignored. `examples/baselines/ppo/runs` is a symlink to `logs/runs` (ppo.py writes `runs/<exp_name>`).
- Env: `source /mmfs1/gscratch/weirdlab/will/venvs/maniskill/bin/activate`; set `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`.
- stdout of `ppo.py` is block-buffered under Slurm; read progress from the TensorBoard events (`EventAccumulator`) or use `python -u`.
- Git: `origin` is the user's fork over SSH, `main` tracks `upstream`; `remote.pushDefault=origin`. The GitHub key `~/.ssh/willh003`
  has a passphrase (needs `ssh-add` in the session).
- Local-driven sync/submit/pull scripts: see `remote-workflow.md` (porting to another repo: `remote-workflow-setup.md`).
