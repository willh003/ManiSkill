# Clutter pick: expert + ICL/BC pilot (2026-11-17)
Env `PickClutterYCBCond-v1` (see `implementations/clutter-cond-env-and-icl.md`). Tracker experiment `clutter-icl`.

## Expert
klone job 43329456 (`slurm/expert_clutter_ppo.sbatch`, 4096 envs, 100M steps, cudagraphs), run `logs/runs/clutter-expert-s1/`.
Eval success_once .73 at ~iter 50, .95-1.00 from iter ~100 on. ckpt_401 (used for the pilots below): .97-.98 attempt-1
success on the 512 eval layouts; per-condition success on bank layouts ~[1.0, .95, .94, 1.0] (banana/brick lowest);
90% of layouts have all 4 conditions succeed (bank keep fraction).
Local attempts died: 4096 envs + cudagraphs OOMed the 30 GB box; eval video OOMed a 14G-capped run; ppo_fast without
cudagraphs needed no_grad on rollout/gae (fixed).

## ICL / BC pilots (200 layouts, seed 1, local, expert ckpt_401; runs `logs/runs/icl-clutter-pilot*-d200-s1`)
attempt-1 success / no-lift, 512 eval layouts:
| run | model | attempt 1 | no-lift | notes |
|---|---|---|---|---|
| pilot | BC flat, target pose visible | .945 | .055 | pass@6 .990 |
| pilot | BC flat, random cond each attempt | .236 | .055 | pass@4 .656 ~ analytic guessing .68 |
| pilot | ICL flat, attn ctx, GMM sticky | .014-.025 | .94 | best-val ckpt + all samplers (step/mean/sticky, temps) same |
| cat | ICL flat, ctx concatenated to obs | .016 | .93 | |
| cat-noqvel | + qvel zeroed | .018 | .94 | not a qvel copycat |
| bconehot | BC flat, only target one-hot (no target pose) | .035 | .93 | **flat MLP can't use an identity** |
| obj | BC object-centric (is-target flag per object token) | .873 | .127 | identity conditioning fixed |
| obj | ICL object-centric | .025 | .89 | |
Oracle contexts on obj-ICL (64 val layouts): right-token .20, 3-wrong-tokens .016, empty 0.0 (all samplers).

Reading: (1) the flat obs needs the target *pose*; an index is not usable from 200 layouts -> object-centric tokens.
(2) With object tokens, ICL still cannot act on an empty/ambiguous context: the per-step GMM never commits to one object
(95% no-lift), and uses an unambiguous context only weakly (.20 vs BC .87).
Also: val NLL bottoms out at 5-10k steps then explodes (log-std collapse); eval now uses the best-val checkpoint.

## Next (user choice: stay end-to-end)
Action chunking (`--chunk 8`, GMM over 8-step chunks, open-loop), 1000 layouts, final expert checkpoint.

## Chunked GMM, 1000 layouts, final expert (`logs/runs/icl-clutter-obj-c8-d1000-s1`)
`--arch obj --chunk 8 --demos 1000 --steps 20000`, expert final_ckpt (attempt-1 .996; bank keep .973).
| model | attempt 1 | pass@4 | pass@6 | no-lift |
|---|---|---|---|---|
| BC obj c8 (target flag) | .691 | .760 | .771 | .31 |
| BC obj c8, random cond | .156 | .486 | .596 | .31 |
| ICL obj c8 (sticky / step / mean) | .025 / .035 / .041 | .061 / .080 / .064 | .076 / .100 / .064 | .85 |
| ICL controls none / shuf_fb | .031 / .025 | .061 / .115 | .062 / .170 | .84 / .75 |
Reading: 8-step open-loop chunks *hurt* BC (.873 -> .691 even with 5x data; 31% no-lift: open-loop grasps miss).
ICL still does not commit with an empty context (~85% no-lift). A GMM head (per step or per chunk) cannot represent the
4-way object choice here: the empty-context ICL problem is pure 4-mode BC and it fails at ~3%.
