# Setting up the local <-> cluster workflow in another repo

Reference implementation: `scripts/cluster/` in this repo; usage is in `remote-workflow.md`.

## Prerequisites
1. **ssh alias with a persistent control socket** in `~/.ssh/config` (Duo/2FA is only needed when the socket expires):
   ```
   Host klone
       HostName klone-login03.hyak.uw.edu
       User wph52
       ControlMaster auto
       ControlPath ~/.ssh/sockets/%C
       ControlPersist 7d
   ```
   `mkdir -p ~/.ssh/sockets`, then run `ssh klone` once and complete Duo. Check with `ssh -o BatchMode=yes klone true`.
2. `rsync` on both ends.
3. A checkout of the repo on the cluster (any rsync target dir works; it does not have to be a git clone) and a venv there.
   If the repo is a Python package, install it editable (`pip install -e .`) in the cluster venv so pushed code is live.

## Steps
1. Copy `scripts/cluster/{config,push,submit,status,pull,cancel}.sh` and `smoke.sbatch` into the new repo's `scripts/cluster/`; `chmod +x scripts/cluster/*.sh`.
2. Edit `config.sh`: set `REMOTE_DIR`; extend `PUSH_EXCLUDES` with the repo's big/local-only dirs (datasets, `.venv`, build outputs).
   Keep `logs/` excluded so pushes never touch results.
3. Edit `smoke.sbatch`: set `--account`/`--partition`, the `--output` path (`<REMOTE_DIR>/logs/slurm/%x-%j.out`; `#SBATCH` lines can't expand variables), the venv path, and the python check.
4. Add `logs/` to `.gitignore`. Real sbatch scripts should write runs under `logs/runs/<name>` so `pull.sh` finds them.
5. Verify, in order:
   - `scripts/cluster/push.sh --dry-run` lists only expected files. If the remote already has hand-synced edits, review this before the first real push.
   - `scripts/cluster/submit.sh scripts/cluster/smoke.sbatch`, then `status.sh`, `status.sh --tail JOBID`, `pull.sh`.
6. Docs for agents: copy `remote-workflow.md` (adjust paths) into the repo's docs and add a line to its `CLAUDE.md` pointing at it.
7. Permissions (user must add these; an agent can't grant itself permissions), in `~/.claude/settings.json` or the repo's `.claude/settings.json`:
   `Bash(scripts/cluster/*)`, `Bash(ssh klone squeue*)`, `Bash(ssh klone sacct*)`.

## Gotchas
- Never run compute on login nodes; scripts only call sbatch/squeue/sacct/scancel/tail.
- `push.sh --delete` removes remote files missing locally; use only when sure.
- `push.sh` sets `--chmod=D755,F644`, which drops the executable bit on remote files (irrelevant for sbatch).
- Python stdout is block-buffered under Slurm; use `python -u` or read TensorBoard events to follow progress.
