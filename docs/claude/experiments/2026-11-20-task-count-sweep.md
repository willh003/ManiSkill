# Number of train holes: PPO vs recurrent PPO (2026-11-20, in progress)

Tracker experiment `task-count-sweep`. All on Tillicum (H200 MIG, state obs), seed 1, `logs/runs/<name>`.

## Setup (identical tasks across all runs)
- Hole pool: `checker_sample(M, 'all', generator seed 10000, wide_corner)`: nested, the first 4 holes are the same for every M.
- Shared peg (size + start pose) is **identical for every M** (drawn as the first of a fixed 4-task draw; commit "shared peg independent of M"). Robot start noise random.
- Held-out: 128 fresh holes (generator seed 777, same distribution). rl2 evaluates 512 holes whose first 128 are those.
- PPO recipe = `ppo_tasks.py` defaults incl. `--near_hole_prob 0.25` (OmniReset-style reset bias; now the default), 2048 envs, 60M steps.
- Recurrent PPO = `rl2_peg.py`, 8 episodes, feedback noise 4 cm xy / 0.2 rad yaw, BC (DAgger) warm start then PPO; evals train + held-out (`held_*` keys).

## Runs
| name | what | job |
|---|---|---|
| p4-scratch-nh25 | PPO M=4 (from the earlier reset-bias experiment) | 504060 |
| ppo-M16/M64/M256-nh25 | PPO, 16/64/256 train holes | 504569-71 |
| rl2-e8-n4cm-M4/16/64/256-s1 | recurrent PPO with BC init | 504568, 504652-54 |
| rl2-e8-n4cm-M4-nobc-nh25-s1 | recurrent PPO, **no BC**, reset bias 0.25 (`--no-bc --near_hole_prob 0.25 --ppo-iters 500`) | 504651 |

## Early observation
rl2 M=4 after 6 BC iterations: train ep2-8 success ~.94-.96 but held-out ~.02 (memorises the 4 holes even with feedback available).
Results: TODO.
