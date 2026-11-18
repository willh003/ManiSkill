# Recurrent PPO upper bound for hidden-hole peg with end-of-episode feedback (2026-11-17)

Plan `ideas/2026-11-17-rl2-peg-upper-bound-plan.md`; code `implementations/rl2-peg.md`. Tracker experiment `rl2-peg-upper-bound`.
Eval = the 512 fixed tasks of the ICL runs (wide_corner 'all', eval_seed 12345), deterministic, same peg / robot start in every episode.

## Scripted references (`meta_peg_oracle.py`, `logs/rl2-peg/oracle-<cfg>.json`)
Success per episode. oracle = ep 1 prior-mean-hole expert, ep k>1 expert on mean_j(f_j + c_j).
| config | expert (true hole) | prior (blind) | oracle |
|---|---|---|---|
| 2 eps, noiseless (e2-n0) | .900 / .898 | .010 / .010 | .012 / .902 |
| 2 eps, noise .5 cm / .025 rad (e2-n05) | | | .012 / .607 |
| 2 eps, noise 1 cm / .05 rad (e2-n1) | | | .012 / .330 |
| 3 eps, noise 1 cm / .05 rad (e3-n1) | .900 / .898 / .904 | | .010 / .332 / .484 (pass@3 .59) |
The clearance is 3 mm, so a 1 cm error in the hole estimate mostly fails for an expert that trusts it.

## Pilot (local, e2-n0, seed 1): `logs/runs/rl2-peg-pilot-s1`
BC (12 DAgger iterations, ~40 min) then 300 PPO iterations (1024 envs, ~25 s/iter on the 5090, ~2.5 h).
- BC init: ep 1 .004, ep 2 .863 (label MSE ep 2 .004, ep 1 .016: ep-1 labels are the prior expert's, so learnable).
- PPO: ep 2 .90 by iteration 30 (= oracle), .955 by 80, **.984 final** (ep 1 .014). PPO beats the expert it was distilled from (.90), since
  it optimises insertion directly (the expert's failures are jams at the hole entrance, see `2026-10-26-peg-expert.md`).
- Controls (ep 2): other task's (f, c) .004; zeroed feedback .002; GRU state reset at the episode boundary .982. The policy reads the explicit
  feedback; the recurrent memory is not needed when (f, c) stay in the observation (expected for noiseless feedback).
- Compare ICL (`2026-11-16-icl-onpolicy-loop.md`, hole hidden, attempt 1 blind): best pass@2 at 2000 demos ~.82-.87; at 100/200 demos far lower.
  The RL upper bound for the same 2-attempt protocol is .98.

## Runs
300 PPO iterations each, 1024 envs. Klone jobs via `scripts/cluster/submit_cmd.sh rl2-peg-<cfg>` (3 seeds in parallel per job, all GPU groups,
first to start cancels the rest): e2-n05 s1-3, e2-n1 s1-3, e3-n1 s1-3. Local (5090, `-loc-` in the name): pilot-s1 (= e2-n0 s1), e2-n0 s2-3,
e2-n1 s1, e3-n1 s1, e2-n2 s1, e2-n4 s1-2, e3-n4 s1. The klone e2-n0 job never started (one weirdlab L40, ckpt did not schedule) and was
cancelled in favour of local seeds. Klone gscratch was very slow on 2026-11-17 evening: jobs on g3118 / g3117 / g3094 sat 15-20 min in
uninterruptible IO importing torch; the g3118 one never recovered and was resubmitted with `SBATCH_EXTRA=--exclude=g3118`.
Checkpoints `logs/runs/rl2-peg-*/checkpoints/{bc.pt,it<k>.pt,latest.pt}`; results `results.json`; plot
`logs/renders/rl2-peg/noise_sweep.png` (`examples/baselines/ppo/rl2_peg_plot.py`, also on the tracker as asset 6da696f2...).

## Results (deterministic eval, 512 tasks; success per episode; ranges over seeds)
Two episodes, success in episode 2 (episode 1 is blind, .002-.016 everywhere):
| feedback noise (xy / yaw) | seeds | BC init | **recurrent PPO** | scripted oracle | zeroed c | shuffled c | GRU reset at boundary |
|---|---|---|---|---|---|---|---|
| none | 3 | .863-.898 | **.982-.990 (mean .986)** | .902 | .000-.002 | .004-.006 | .979-.982 |
| .5 cm / .025 rad | 3 | .805-.875 | **.975-.984 (.980)** | .607 | .000-.006 | .006-.010 | .969-.980 |
| 1 cm / .05 rad | 4 | .762-.803 | **.967-.973 (.970)** | .330 | .000 | .006-.020 | .955-.969 |
| 2 cm / .1 rad | 1 | .574 | **.910** | .203 | .002 | .008 | .902 |
| 4 cm / .2 rad | 2 | .303-.332 | **.668-.674 (.671)** | .068 | .000 | .018-.020 | .637-.713 |

Three episodes (ep 2 / ep 3):
| noise | seeds | BC init | **recurrent PPO** | pass@3 | oracle | zeroed c | GRU reset at boundaries |
|---|---|---|---|---|---|---|---|
| 1 cm | 4 | .730-.809 / .828-.867 | **.971-.979 / .980-.990** | .98-.99 | .332 / .484 | <= .006 | .957-.980 / .977-.984 |
| 4 cm | 1 | .381 / .424 | **.656 / .836** | .848 | .053 / .125 | .002 / .006 | .646 / **.723** |

## Reading
- **Upper bound for the ICL setting (2 episodes, exact feedback c = hole - f): ~.986 success on the attempt after feedback**, vs .90 for
  the expert with the true hole and for the scripted oracle (expert on f + c). PPO improves on the expert it was distilled from (the
  expert's failures are jams at the hole entrance, `2026-10-26-peg-expert.md`). The hidden-hole ICL transformer's best pass@2 (2000 demos)
  is ~.82-.87 and much lower at 100 / 200 demos (`2026-11-16-icl-onpolicy-loop.md`), so there is a .1-.6+ gap to close.
- **It uses the feedback**: zeroed or another task's (f, c) drops episode 2 to the blind level (<= .02) in every config.
- **Noise up to ~2 cm costs little** (.986 -> .980 -> .970 -> .910 at 0 / .5 / 1 / 2 cm), while the oracle that trusts the noisy estimate
  collapses (.90 -> .61 -> .33 -> .20). The policy goes to the feedback's estimate and corrects closed-loop from contact (peg / TCP
  states react when the peg hits the box face). At 4 cm it reaches .67 (oracle .07). A Bayes-optimal ICL policy should show the same
  robustness; the scripted oracle is not the right upper bound once feedback is noisy.
- **Memory**: with one feedback in the observation the GRU state is not needed (reset at the boundary changes nothing, also for 3
  episodes at 1 cm, where one feedback + contact search already gives .97). At 4 cm with 3 episodes the policy combines the two feedbacks:
  ep 3 .836 vs ep 2 .656, and resetting the GRU at the boundaries cuts ep 3 to .723 (ep 2 unchanged). So memory over episodes matters only
  once single feedbacks are uninformative.
- Episode 1 stays blind (~.01): nothing in this reward makes ep-1 exploration pay; ep-1 behaviour is the BC prior (aim at the mean hole).
- BC init (DAgger from the scripted oracle, expert labels) is already strong (.86-.90 noiseless) and PPO converges in 100-300 iterations.

## Possible next steps
- Give the ICL transformer the same noisy feedback to see how close it gets to these bounds (the ICL runs so far are noiseless).
- Make ep-1 exploration matter (e.g. weaker feedback such as a direction or a distance only, or reward only on the last episode) to test
  whether RL learns informative first attempts.
