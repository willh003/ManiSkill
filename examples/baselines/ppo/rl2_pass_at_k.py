"""Per-task success matrix + pass@k of one rl2_peg checkpoint. Usage: python rl2_pass_at_k.py RUN_DIR CKPT_NAME OUT.json"""
import json, sys, torch
from rl2_peg import Args, Trainer
run, ck, out = sys.argv[1:4]
a = json.load(open(f"{run}/config.json"))["args"]
a["num_envs"] = 1024
import glob; a["expert_checkpoint"] = "logs/runs/peg-expert-r1/ckpt_3576.pt"
tr = Trainer(Args(**a))
tr.actor.load_state_dict(torch.load(f"{run}/checkpoints/{ck}", map_location=tr.dev)["actor"])
S = tr.run_episode("student", goals=tr.eval_goals, seed=a["eval_seed"])["succ"][:512].int().cpu()
json.dump(dict(succ_matrix=S.tolist(), perf=S.float().mean(0).tolist(), pass_at=[S[:, :k].amax(1).float().mean().item() for k in range(1, S.shape[1] + 1)]), open(out, "w"))
print(run, ck, "perf", [round(v, 3) for v in S.float().mean(0).tolist()], "pass", [round(S[:, :k].amax(1).float().mean().item(), 3) for k in range(1, S.shape[1] + 1)])
