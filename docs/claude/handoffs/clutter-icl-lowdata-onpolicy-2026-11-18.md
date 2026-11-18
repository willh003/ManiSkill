# Handoff: clutter ICL, low-data BC + on-policy rounds (2026-11-18)

Background: `docs/claude/implementations/clutter-cond-env-and-icl.md`, results so far `docs/claude/experiments/2026-11-17-clutter-icl-pilot.md`.

## User request
1. Run the same BC-vs-ICL experiments with less data, so BC given the target is not perfect (~80%).
2. Train the ICL student on its own attempts (multi-round on-policy loop) and see if it helps.

## State
- Code: `examples/baselines/ppo/icl_clutter.py`, `--rounds R --p-pool 0.5`. Each round: roll out the student 6 attempts on every (demo layout, target) with real feedback, add the sequences to the pool, retrain from scratch with half of each batch's contexts from the pool. Labels are still the expert demo for the true target.
- Commit 29e8c76 fixed a crash (last pool chunk smaller than num_envs) and made `--load-dir` resume: load existing ckpts, train the missing ones.
- Klone jobs 43338727.. all FAILED at round-1 pool collection. Their round-0 run dirs were pulled to `logs/klone-r0/icl-clutter-diff-d{50,100,200,1000}-r4-s1`.
- **Running locally**: queue `logs/queues/clutter-onpolicy-local.sh`, run by xargs -P1 in the order d1000, 200, 300, 100, 50.
  - systemd scope with a 10G memory cap; each job uses ~4 GB RSS.
  - Another session's rl2_peg jobs share the GPU and RAM, so the queue runs one job at a time.
  - Logs: `logs/runs/icl-clutter-diff-d<N>-r4-s1.local.log`. Run dirs: `logs/runs/icl-clutter-diff-d<N>-r4-s1/` (`results.json` has `rounds`).
  - d300 has no klone round 0, so it trains from scratch.
  - Each round takes ~2.5 min to train plus pool and eval time.
- Tracker docs: `runs/icl-clutter-diff-d{50,100,200,300,1000}-r4-s1`. Update status, lastMetrics and images when done.

## Round-0 results (attempt-1 BC, pass@k from 512 eval layouts)
| demos | BC succ | guess pass@4 | ICL pass@4 | ICL pass@6 |
|---|---|---|---|---|
| 50 | .31 | .30 | .02 | .03 |
| 100 | .62 | .49 | .07 | .11 |
| 200 | .73 | .59 | .18 | .23 |
| 1000 | .97 | .70 | .52 | .68 |

- ICL degrades far faster than BC with less data; mostly it fails to lift anything (80-98% no-lift at ≤200 demos).
- At 1000 demos ICL uses feedback (perf@k rises .15→.57, while the none/shuf_fb controls stay flat), but it is still below guessing at pass@4.
- d1000 round-1 pool: per-attempt success on the demo tasks is .14/.24/.34/.48/.54/.60.

## Next steps
1. When the queue finishes, compare per-round pass@k/perf@k (rounds 0-4) against BC, guess and oracle for each demo count. Check whether on-policy rounds cut the no-lift rate.
2. Extend `icl_clutter_plot.py` to plot rounds. Write an experiments note. Update the tracker (done, metrics, images).
3. git push needs the user to run `! ssh-add`; commits are local only.
