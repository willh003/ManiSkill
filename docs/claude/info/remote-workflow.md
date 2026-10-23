# Local <-> klone workflow (agent-driven)

Edit locally, rsync to klone, run via Slurm, pull results back. All scripts are in `scripts/cluster/` and use
`ssh klone` (ControlMaster socket, so no prompts). Settings (host, remote dir, excludes) live in `scripts/cluster/config.sh`.

| Step | Command |
|---|---|
| Preview sync | `scripts/cluster/push.sh --dry-run` (content diff only; `--delete` also removes remote files missing locally) |
| Sync | `scripts/cluster/push.sh` |
| Sync + sbatch | `scripts/cluster/submit.sh examples/baselines/ppo/slurm/<x>.sbatch [sbatch args]` |
| Queue / recent jobs | `scripts/cluster/status.sh` |
| Job output | `scripts/cluster/status.sh --tail JOBID [N]` |
| Fetch results | `scripts/cluster/pull.sh [--no-ckpt] [run-name ...]` (no names = all of `logs/`) |
| Cancel | `scripts/cluster/cancel.sh JOBID...` |
| Smoke test | `scripts/cluster/submit.sh scripts/cluster/smoke.sbatch` |

Loop: edit -> `submit.sh` -> `status.sh` -> `pull.sh` -> write `docs/claude/experiments/` note.

Notes
- Local tree is the source of truth. Push is one-way rsync (checksum-based, git is not involved). `logs/` is never pushed or deleted remotely.
- Remote path `/mmfs1/gscratch/weirdlab/will/ManiSkill`; `mani_skill` is an editable install there, so pushed code is live with no reinstall.
- Sbatch `--output` hardcodes the remote path (`#SBATCH` lines can't expand variables).
- Only manual step: if scripts print "cannot reach klone non-interactively", the user must run `! ssh klone` (Duo) to refresh the socket.
- Never run compute on login nodes; only sbatch/squeue/sacct/scancel/tail go over ssh.
