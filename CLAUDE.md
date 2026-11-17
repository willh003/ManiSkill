If you want documentation on a specific part of the implementation, check docs/, especially docs/claude

Whenever you do a large or important implementation, save information about it in organized markdown files within docs/claude. Make sure to keep this files clean and organized for yourself. You should keep the following structure in your docs
- experiments
- handoffs (for other agents)
- ideas
- implementations
- info

Whenever you run an experiment, write a brief description of what you did and the results in experiments/. This should include references to plots, checkpoints, and other relevant info in other directories. 

Keep this repo clean - periodically clean up files, dead experiments, etc.

Make sure large files aren't committed by modifying .gitignore. Keep runs and experiment outputs in logs/ (which should not be committed).


You're working on a project that aims to pretrain policies capable of solving POMDPs, so they are better equipped to solve epistemic POMDPs at test time by searching for the solution.
This involves taking an existing MDP, making it partially observable, and providing feedback to a memory-conditioned policy, so it can learn to search for the answer given its history and the corrective feedback.



You will either be running from within hyak klone, a large shared compute cluster, or from a local machine with a 5090 GPU.
- If you are local, you can do quick python tests and do small/quick runs locally. You should submit larger experiments or parallel runs to hyak klone by syncing the code there, then submitting jobs from there, then syncing results back here
- If you are on the cluster you can run on l40 gpus (preferably) or l40s/a40 gpus. You should start by using the partition group weirdlab, then use ckpt. Info is provided in docs/claude/info/cluster-and-logging.md


To run on klone from the local machine (push, sbatch, status, pull), see docs/claude/info/remote-workflow.md (scripts/cluster/).


## Working rules (added 2026-11-10)
- **Commit code changes.** Whenever you modify or add code (scripts, configs, docs), commit it promptly (small, descriptive commits) before launching runs that depend on it, so each run's git_commit in config.json is meaningful. Don't leave experiment scripts untracked/dirty.
- **Report runs to the experiment-tracking artifact** (https://claude.ai/artifact/3trmWhwWtEXCXZmKsrw1qh) with the ArtifactData tool (load it with ToolSearch if deferred), for every run you launch:
  1. At launch: action "set", collection "runs", doc_id "<run-id>", data {name, experimentId, agent, host, status:"running", startedAt, heartbeat, updated (ISO times), config:{...}, summary:"one line on what the run does", tags:[...]}.
  2. Every few minutes while running: action "update" on runs/<run-id> with {heartbeat, updated, lastMetrics:{...}} (pin if_version).
  3. Curves: action "set" on collection "metrics", doc_id "<run-id>", data {series:{name:[[x,y],...]}} (resend full series, <~2000 points).
  4. Images: upload with the Artifact tool (asset:true, url=<artifact url>, file_path=plot.png), then update runs/<run-id> {images:[{asset:"<id>", caption:"..."}]}.
  5. When finished: update status to "done" or "failed" plus summary.
  Optional: experiments/<exp-id> {name, hypothesis, tags, notes}; agents/<name> {name, host, task, status, heartbeat}.
- Launch long jobs detached (`setsid nohup ... &`) so they survive the session ending; background tasks started from the session can be killed when it ends.
