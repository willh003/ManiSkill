# PegInsertionSide-v1 PPO expert (2026-10-26)

Goal: state-based expert for the clean task, to be the teacher for hidden / biased hole-pose ICL experiments (see `2026-10-24-pickcube-bias-icl.md`).

## Runs
| job | script | outcome |
|---|---|---|
| 42340023 | `ppo.py`, 2048 envs, 100 steps/rollout, gamma 0.97 | crashed at 37M steps (`Disk quota exceeded`, group file quota 99%); eval success_once 0.0 throughout. Discarded. |
| 42345612, 42351168 | `ppo_fast.py` | import errors (klone venv lacked `tensordict`, `wandb`) |
| 42352113 | `slurm/expert_peg_ppo.sbatch` (`ppo_fast.py`, official baseline recipe: 2048 envs, 16 steps, gamma 0.97, lambda 0.95, 8 epochs, 32 minibatches, 75M steps, cudagraphs) | finished, ~6 h on L40 |

Checkpoints: `logs/runs/peg-expert-fast/final_ckpt.pt` (and `ckpt_*.pt`); same state-dict layout as `ppo.py`'s Agent.

## Result
Deterministic eval, 256 envs, 100 steps, 2 seeds: **success_once 0.73 / 0.74**. In-training eval (16 envs) fluctuates 0.25-0.94.
(An earlier version of this note claimed the policy withdraws the peg: wrong. `success_at_end` read 0.0 only because the env auto-resets at the last step and `info` came from the reset state; at the last pre-reset step success is 0.727 = success_once, and successful episodes stay successful for 97.5% of post-first-success steps, median first success at step 36.)
Failure mode (27%): grasped and aligned (peg-head yz distance to the hole axis ~4-5 mm; clearance is 3 mm) but stuck at the hole entrance, head ~10 cm short of the hole centre (jammed). Needs finer alignment, i.e. more training / sharper alignment reward.
Eval script was a scratch file (not committed); re-create from `ppo_fast.py` Agent if needed.

## Notes
- The first `ppo.py` config (100-step rollouts) learned nothing in 37M steps; the `ppo_fast.py` recipe works.
- Klone venv: `tensordict==0.6.2` (0.7+/latest need newer libstdc++ / torch), `wandb` installed. An accidental `uv pip install tensordict` upgraded torch to 2.14; reverted to torch 2.6.0, triton 3.2.0 (orphan `nvidia-*-13` packages left behind).
- Expert is weaker than the PickCube one (0.99); the student ceiling is ~0.73 unless the expert is trained longer. Next: resume runs, see below.
