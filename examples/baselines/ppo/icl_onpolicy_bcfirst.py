"""Re-evaluate an icl_onpolicy.py run with attempt 1 played by the run's own visible-hole BC (tf-bc.pt, deterministic), then the
ICL model of each round for attempts 2..K (sliding window of one attempt, the first context is the BC rollout).
Same 512 eval tasks as the original eval. Controls at the last round: `randf` (another task's attempt-1 rollout, exact goal) and
`fixed` (the BC attempt-1 context reused for every retry).
Usage: python icl_onpolicy_bcfirst.py RUN_DIR [RUN_DIR ...]   -> RUN_DIR/results_bcfirst.json (same layout as results.json rounds)"""
import json
import sys

import gymnasium as gym
import torch

from icl_checker import rollout
from icl_tasks import checker_sample
from icl_transformer import LAYOUT, PEG, TF, final_idx, hist_of, mk
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo import Agent


class A_:  # rollout() reads num_envs / num_steps from here
    pass


def main(run_dirs):
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg0 = json.load(open(f"{run_dirs[0]}/config.json"))["args"]
    args = A_()
    args.num_envs, args.num_steps = cfg0["num_envs"], cfg0["num_steps"]
    N, K = args.num_envs, cfg0["K"]
    env = gym.make("PegInsertionSideBias-v1", num_envs=N, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
    env = FlattenActionSpaceWrapper(env) if isinstance(env.action_space, gym.spaces.Dict) else env
    venv = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)
    A = venv.single_action_space.shape[0]

    class _S:
        single_observation_space = type("S", (), {"shape": (43,)})
        single_action_space = venv.single_action_space

    expert = Agent(_S).to(dev)
    expert.load_state_dict(torch.load(cfg0["expert_checkpoint"], map_location=dev))
    expert.eval()
    ar = torch.arange(N, device=dev)
    ones = torch.ones(N, device=dev)
    fin = lambda r: r["obs"][ar, final_idx(r["alive"])][:, PEG]
    goals = checker_sample(N, "all", dev, torch.Generator(device=dev).manual_seed(cfg0["eval_seed"]), cfg0["split"])
    ex_peg = fin(rollout(venv, expert, None, args, goals, cfg0["eval_seed"], "expert"))

    def ctx_of(y, valid):
        return dict(s=torch.zeros(N, 1, 43, device=dev), a=torch.zeros(N, 1, A, device=dev), y=y.reshape(N, 1, -1),
                    valid=torch.full((N, 1), valid, dtype=torch.bool, device=dev))

    for rd in run_dirs:
        cfg = json.load(open(f"{rd}/config.json"))["args"]
        kind, L, R = cfg["kind"], LAYOUT[cfg["kind"]], cfg["rounds"]
        bc = TF(A, 0, cfg["d"], cfg["layers"], cfg["heads"]).to(dev)
        bc.load_state_dict(torch.load(f"{rd}/checkpoints/tf-bc.pt", map_location=dev))
        bc.eval()
        r1 = rollout(venv, expert, bc, args, goals, cfg["eval_seed"], "student", ctx_of(torch.zeros(N, 7, device=dev), False), "normal", mask_pose=False)
        f1, h1 = fin(r1), hist_of(r1["obs"], final_idx(r1["alive"]))
        out = dict(bc_attempt1=r1["succ"].float().mean().item(), rounds={})
        for rnd in range(1, R + 1):
            pol = TF(A, L, cfg["d"], cfg["layers"], cfg["heads"], 0.0, True).to(dev)
            pol.load_state_dict(torch.load(f"{rd}/checkpoints/round{rnd}.pt", map_location=dev))
            pol.eval()
            res = {}
            for mode in ("real", "fixed", "randf") if rnd == R else ("real",):
                r, succ = r1, [r1["succ"]]
                for k in range(1, K):
                    if mode == "real":
                        f, h = fin(r), hist_of(r["obs"], final_idx(r["alive"]))
                    elif mode == "fixed":
                        f, h = f1, h1
                    else:
                        pm = torch.randperm(N, device=dev)
                        f, h = f1[pm], h1[pm]
                    r = rollout(venv, expert, pol, args, goals, cfg["eval_seed"], "student", ctx_of(mk(kind, h, f, ex_peg, ones), True), "normal", mask_pose=True)
                    succ.append(r["succ"])
                s = torch.stack(succ, 1)
                pre = "" if mode == "real" else mode + "_"
                res[pre + "succ"] = s.float().mean(0).tolist()
                res[pre + "pass_k"] = s.cummax(1).values.float().mean(0).tolist()
            out["rounds"][str(rnd)] = res
            json.dump(out, open(f"{rd}/results_bcfirst.json", "w"), indent=1)
            print(f"{rd.rstrip('/').split('/')[-1]} round {rnd}: pass@k {[round(x, 3) for x in res['pass_k']]}", flush=True)
        out["done"] = True
        json.dump(out, open(f"{rd}/results_bcfirst.json", "w"), indent=1)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
