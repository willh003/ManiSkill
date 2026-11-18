"""Scripted references on MetaPeg (512 fixed eval tasks, wide_corner 'all', eval_seed 12345, same as icl_onpolicy evals).
  expert : expert with the true hole in every episode (ceiling)
  prior  : expert told the prior-mean hole in every episode (blind)
  oracle : episode 1 = prior expert, episode k>1 = expert on the hole estimate = mean of f_j + c_j over the finished episodes
           (= true hole with noiseless feedback; posterior mean under a flat prior with iid Gaussian noise)
Usage: python meta_peg_oracle.py [--n-ep 2] [--fb-noise-xy 0] [--fb-noise-yaw 0] [--out file.json]"""
import json
from dataclasses import dataclass

import torch
import tyro

from icl_tasks import checker_sample
from meta_peg import MetaPeg, believed_hole_obs, load_expert, prior_hole_obs
from run_logging import LOG_ROOT


@dataclass
class Args:
    expert_checkpoint: str = str(LOG_ROOT / "runs/peg-expert-r1/ckpt_3576.pt")
    num_envs: int = 512
    n_ep: int = 2
    fb_noise_xy: float = 0.0
    fb_noise_yaw: float = 0.0
    eval_seed: int = 12345
    split: str = "wide_corner"
    out: str = ""


def run(env, expert, mode, goals, seed):
    env.reset(seed, goals)
    done = False
    while not done:
        o = env.expert_obs()
        if mode == "expert":
            x = o
        elif mode == "prior" or env.ep == 0:
            x = prior_hole_obs(o)
        else:
            est = torch.stack([f + c for f, c in zip(env.f_hist, env.c_hist)]).mean(0)
            x = believed_hole_obs(o, est)
        with torch.no_grad():
            a = expert.get_action(x, deterministic=True)
        *_, done, _ = env.step(a)
    s = env.succ.float()
    return dict(succ=s.mean(0).tolist(), pass_k=s.cummax(1).values.mean(0).tolist())


def main(args):
    dev = torch.device("cuda")
    env = MetaPeg(args.num_envs, args.n_ep, args.fb_noise_xy, args.fb_noise_yaw, args.split, device="cuda")
    expert = load_expert(args.expert_checkpoint, dev)
    goals = checker_sample(args.num_envs, "all", dev, torch.Generator(device=dev).manual_seed(args.eval_seed), args.split)
    res = {m: run(env, expert, m, goals, args.eval_seed) for m in ("expert", "prior", "oracle")}
    for m, r in res.items():
        print(m, "succ", [round(x, 3) for x in r["succ"]], "pass@k", [round(x, 3) for x in r["pass_k"]])
    if args.out:
        json.dump(dict(args=vars(args), **res), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main(tyro.cli(Args))
