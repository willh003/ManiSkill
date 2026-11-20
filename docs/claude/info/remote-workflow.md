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

## Tillicum (UW Hyak, H200) — `CLUSTER=tillicum` (full note: tillicum.md)
Same scripts, different profile: `CLUSTER=tillicum scripts/cluster/{push,submit,status,pull,cancel}.sh ...` (default is klone).
`submit_cmd.sh` also works: `CLUSTER=tillicum TIME=8:00:00 scripts/cluster/submit_cmd.sh <name> "<cmd>"` (one job, no ckpt fallback).
- ssh alias `tillicum` (tillicum.hyak.uw.edu, user wph52, control socket; first login needs `! ssh tillicum` + Duo).
- Repo: `/gpfs/projects/weirdlab/will/ManiSkill`; venv `/gpfs/projects/weirdlab/will/venvs/maniskill` (py3.11, torch 2.6.0+cu124, editable mani_skill).
- Account `weirdlab`, partition `gpu-h200-mig` (g024, 56 x `h200_1g.18gb` slices; **max 2 CPUs per GPU**, 30G/CPU default mem).
  Full `gpu-h200` partition exists too (not set up here).
- **No rendering on MIG**: NVIDIA Vulkan is unsupported on MIG slices (`ErrorIncompatibleDriver`), and sapien still needs *a* Vulkan device
  to build materials. Workaround (set in `cmd_tillicum.sbatch`): `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`
  (lavapipe, CPU) + `MANISKILL_RENDER_BACKEND=none` (env override in `sapien_env.py`). State-obs GPU sim works; camera/rgb obs does not.
  Verified with `scripts/cluster/smoke_tillicum.sbatch` (tries every ICD/backend combo).
- Compute nodes reach the internet through a proxy; sapien downloads the PhysX GPU lib to `~/.sapien` on first use (done).
- Never use `find /` or heavy scans in jobs/login: GPFS is slow.
