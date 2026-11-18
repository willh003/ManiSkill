# What memory does the recurrent PPO policy use? (2026-11-18)
Script `examples/baselines/ppo/rl2_memory.py RUN_DIR OUT.json` (`PROBES_ONLY=1` skips interventions); results `logs/rl2-memory/e3-n4-s1{,-probes}.json`.
Policy: `rl2-peg-e3-n4-loc-s1` (3 episodes, 4 cm noise, latest.pt), 512 eval tasks, deterministic, fresh feedback noise (so ~+-.03 vs training eval).
The actor obs holds only the LATEST feedback (f, c), so ep-1 feedback reaches ep 3 only through the GRU.

## Interventions (success ep1 / ep2 / ep3)
base .012/.713/.809. Reset GRU at all boundaries .654/.705; only at ep3 start ep3 .750; swap in another task's GRU at ep3 start ep3 .652.
Zero or shuffle observed (f,c) in ep k: that episode -> ~0 (.000-.02 ep2, .004-.02 ep3); GRU cannot substitute.
Reset GRU every K steps (K=1,2,5,10,25,50): ep2 .31/.37/.41/.52/.59/.64, ep3 .38/.43/.54/.58/.65/.75.
## Probes (ridge, held-out R2, 3x1024 tasks)
At ep3 start: noise of ep-1 estimate (est1 - true hole, xy): GRU .47 vs obs .05; est1-est2: GRU+obs .66 vs obs .24. Gone by step 30 (.08).
At ep2 start the GRU holds nothing extra (obs has ep-1 feedback). GRU state is a worse code of current feedback than the obs itself.
## Reading
1. The observed correction is what the policy acts on; memory cannot stand in for it.
2. Biggest GRU use is short-term within-episode history (reset every few steps costs .2-.4).
3. Cross-episode memory exists (ep-1 noise decodable at ep3 start, wrong memory costs .16) but is used only as a modest, early-episode refinement (~.06 gain).
Caveats: one seed/checkpoint; linear probes show availability, interventions show use. Not yet run: 1 cm 3-ep and 2-ep 4 cm checkpoints.
