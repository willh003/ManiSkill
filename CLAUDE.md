If you want documentation on a specific part of the implementation, check docs/, especially docs/claude

Whenever you do a large or important implementation, save information about it in organized markdown files within docs/claude. Make sure to keep this files clean and organized for yourself. You should keep the following structure in your docs
- experiments
- handoffs (for other agents)
- ideas
- implementations
- info

Whenever you run an experiment, write a brief description of what you did and the results in experiments/. This should include references to plots, checkpoints, and other relevant info in other directories. 

Keep this repo clean - periodically clean up files, dead experiments, etc.

For training runs, you can run on l40 gpus (preferably) or l40s/a40 gpus. You should start by using the partition group weirdlab, then use ckpt

You're working on a project that aims to pretrain policies capable of solving POMDPs, so they are better equipped to solve epistemic POMDPs at test time by searching for the solution.
This involves taking an existing MDP, making it partially observable, and providing feedback to a memory-conditioned policy, so it can learn to search for the answer given its history and the corrective feedback.