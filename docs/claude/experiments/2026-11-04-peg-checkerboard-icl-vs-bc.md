# Peg checkerboard hold-out: ICL (per-step feedback) vs BC (2026-11-04)

Implementation: `implementations/checkerboard-holdout.md`. Question: does ICL pass@2 on held-out black cells beat BC pass@2?
Runs (klone, l40, 30 iterations, 512 tasks, 2000 grad steps/iter): jobs 42837410-12 (`logs/runs/checker-icl-s{1,2,3}`), 42837413-15 (`checker-bc-s{1,2,3}`).
Smoke test locally: expert 0.98 white / 1.00 black, wrong-pose expert 0.05.

Results: pending.
