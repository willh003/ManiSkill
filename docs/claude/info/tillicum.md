# Tillicum (UW Hyak H200 cluster) — how to run ManiSkill jobs here

Set up 2026-11-19. Same local-driven workflow as klone (`remote-workflow.md`), selected with `CLUSTER=tillicum`.

## Quick use (from the local machine)
```
export CLUSTER=tillicum
scripts/cluster/push.sh | submit.sh <sbatch> | status.sh [--tail JOBID] | pull.sh [run...] | cancel.sh JOBID
TIME=14:00:00 scripts/cluster/submit_cmd.sh <job-name> "python -u ppo_tasks.py ... --out_dir \$ROOT/logs/runs/<name>"
```
`submit_cmd.sh` runs `cmd_tillicum.sbatch`: ONE job (no ckpt fallback / sibling cancel as on klone), cwd `examples/baselines/ppo`,
`$ROOT` = repo, output in `logs/slurm/<name>.log` (+ `<name>-<jobid>.out`). Default time 4h (override with TIME).
To move a queued klone job: copy its command from the running sibling's `logs/slurm/<name>-<id>.out` (`+ ...` line) and resubmit; klone's copy keeps
running/queued unless you cancel it (duplicate runs write to separate clusters' `logs/runs/<name>`).

## Facts
- ssh alias `tillicum` -> tillicum.hyak.uw.edu, user wph52, control socket (first login / expiry: user runs `! ssh tillicum`, Duo).
- Repo `/gpfs/projects/weirdlab/will/ManiSkill`; venv `/gpfs/projects/weirdlab/will/venvs/maniskill` (uv, py3.11, torch 2.6.0+cu124,
  tensordict 0.6.2, editable mani_skill). Installing: `uv` is at /usr/local/bin/uv; login nodes have internet via proxy. If a script hits
  ModuleNotFoundError, compare with klone's venv (`ls .../site-packages | grep dist-info`) and `uv pip install` it.
- Slurm: account `weirdlab`, partition `gpu-h200-mig` (node g024, 56 slices `h200_1g.18gb` = 1/7 H200, 18GB). `--gres=gpu:1`;
  **max 2 CPUs per GPU** (sbatch rejects more); DefMemPerCPU 30G. Partition `gpu-h200` (full H200s, 23 nodes) exists, unused so far.
- Default `/gpfs` is slow for scans: never `find /` (a probe job hung on it until time limit).
- Only cancel specific job ids (`scancel <id>`); don't `scancel -u`.

## Rendering is impossible on MIG slices
NVIDIA Vulkan doesn't support MIG (`vk::createInstanceUnique: ErrorIncompatibleDriver`). SAPIEN still needs some Vulkan device to build
URDF `RenderMaterial`s, so even `render_backend=none` fails ("failed to find a rendering device") with the NVIDIA ICD. What works (verified,
`scripts/cluster/smoke_tillicum.sbatch` tries all combos): `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json` (lavapipe CPU ICD)
+ `render_backend="none"`. `cmd_tillicum.sbatch` exports both; the `MANISKILL_RENDER_BACKEND` env var is read in `sapien_env.py` because the
training scripts don't pass render_backend. State-obs GPU sim only (PPO, RL2 peg, etc.); no rgb/pcd/camera obs or video capture.
For rendering use klone or try the full `gpu-h200` partition (untested).

## Gotchas seen
- 2048 envs + cudagraphs on an 18GB slice: watch for OOM in the `.log`; reduce `--num_envs` if so.
- Python stdout is block-buffered: use `python -u`.
