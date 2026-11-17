"""Does the ICL goal encoding lose information? Single-attempt BC on the expert demos, varying how the goal reaches the policy.
Same demos (seed * 100000), val demos, eval tasks, model and optimiser as icl_onpolicy.py (docs/claude/experiments/2026-11-17-icl-goal-encoding.md).

 hole_obs : hole pose in obs dims 35:42 (= the visible-hole BC of icl_onpolicy round 0)
 g_obs    : hole hidden, obs dims 35:42 replaced by g = the expert demo's final peg pose (flag 0)
 hole_cat : hole hidden, hole pose given through the ICL concat context pathway (L=1)
 g_cat    : hole hidden, g through the concat pathway (= ICL [f+c] with the exact goal)
 fc_cat   : hole hidden, [f, c = g - f] through the concat pathway, f = another task's g (as ICL [f, c] with a far-off f)
 fc_failrel: as fc_fail, but f = g - c_fail with c_fail = a failure's error against its own task's goal (realistic feedback size
            and direction, as at ICL test time where f is the same task's previous attempt)
 fc_fail  : as fc_cat, but f = a failed final peg pose drawn from an icl_onpolicy pool (--fail-pool state.pt), redrawn for every
            training sample; val and eval use one fixed draw per task. Same inputs as ICL [f, c], minus the on-policy loop.
Deterministic policy, one attempt on the 512 eval tasks. -> RUN_DIR/results.json {variant: {succ, val, loss}}"""
import json
import math
import os
from dataclasses import dataclass

import gymnasium as gym
import torch
import torch.nn as nn
import tyro

from icl_checker import POSE, apply_mask, rollout
from icl_tasks import checker_sample
from icl_transformer import PEG, TF, final_idx
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo import Agent
from run_logging import LOG_ROOT, RunDir

VARIANTS = ("hole_obs", "g_obs", "hole_cat", "g_cat", "fc_cat", "fc_fail", "fc_failrel")


@dataclass
class Args:
    expert_checkpoint: str = str(LOG_ROOT / "runs/peg-expert-r1/ckpt_3576.pt")
    out_dir: str = str(LOG_ROOT / "runs/icl-goalrep")
    demos: int = 100
    val_demos: int = 200
    variants: str = ",".join(VARIANTS[:5])  # fc_fail only on request (needs --fail-pool)
    num_envs: int = 512
    num_steps: int = 100
    steps: int = 10000
    warmup: int = 500
    batch_size: int = 128
    lr: float = 3e-4
    d: int = 128
    layers: int = 2
    heads: int = 4
    seed: int = 1
    fail_scale: float = 1.0  # fc_failrel: failure error vectors multiplied by this (0 -> f = g, c = 0; checks the pipeline)
    amp: bool = True  # bf16 autocast in training (as icl_onpolicy); --no-amp = fp32
    fail_pool: str = ""  # fc_fail: state.pt of an icl_onpolicy run; its failed rollouts' final peg poses
    eval_seed: int = 12345
    split: str = "wide_corner"


class Wrap(nn.Module):
    """rollout() adapter: builds the per-env goal input from fixed per-env tensors (hole pose h, goal g, distractor f)."""

    def __init__(self, tf, variant, hole, g, f):
        super().__init__()
        self.tf, self.v, self.hole, self.g, self.f = tf, variant, hole, g, f

    def encode(self, *a):
        return None, None

    def decode(self, x, memory, pad):
        n = x.shape[0]
        return self.tf(*inputs(self.v, x, self.hole, self.g, self.f, n))


def inputs(v, obs, hole, g, f, n):
    """obs (n,T,43) with the true hole pose -> (query obs, context (n,L,7) or None, valid)."""
    valid = torch.ones(n, dtype=torch.bool, device=obs.device)
    if v == "hole_obs":
        return obs, None, None
    q = apply_mask(obs, valid)
    if v == "g_obs":
        q = q.clone()
        q[..., POSE], q[..., 42] = g[:, None].expand_as(q[..., POSE]), 0.0
        return q, None, None
    c = {"hole_cat": lambda: hole[:, None], "g_cat": lambda: g[:, None]}.get(v, lambda: torch.stack([f, g - f], 1))()
    return q, c, valid


def main():
    args = tyro.cli(Args)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    run = RunDir(args.out_dir, args)
    torch.manual_seed(args.seed)
    N = args.num_envs
    env = gym.make("PegInsertionSideBias-v1", num_envs=N, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
    env = FlattenActionSpaceWrapper(env) if isinstance(env.action_space, gym.spaces.Dict) else env
    venv = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)
    A = venv.single_action_space.shape[0]

    class _S:
        single_observation_space = type("S", (), {"shape": (43,)})
        single_action_space = venv.single_action_space

    expert = Agent(_S).to(dev)
    expert.load_state_dict(torch.load(args.expert_checkpoint, map_location=dev))
    expert.eval()

    def collect(n_need, seed0):
        ob, lb, al, have, b = [], [], [], 0, 0
        while have < n_need:
            goals, seed = checker_sample(N, "all", dev, None, args.split), seed0 + b * 20
            r = rollout(venv, expert, None, args, goals, seed, "expert")
            ok = r["succ"].nonzero()[:, 0][: n_need - have]
            ob.append(r["obs"][ok]), lb.append(r["lab"][ok]), al.append(r["alive"][ok])
            have, b = have + len(ok), b + 1
        return torch.cat(ob), torch.cat(lb), torch.cat(al)

    EO, EL, EA = collect(args.demos, args.seed * 100000)
    VO, VL, VA = collect(args.val_demos, args.seed * 100000 + 30000)
    eval_goals = checker_sample(N, "all", dev, torch.Generator(device=dev).manual_seed(args.eval_seed), args.split)
    ex = rollout(venv, expert, None, args, eval_goals, args.eval_seed, "expert")
    peg = lambda O, Al: O[torch.arange(len(O)), final_idx(Al)][:, PEG]
    E, NV = len(EO), len(VO)
    eg, vg, xg = peg(EO, EA), peg(VO, VA), peg(ex["obs"], ex["alive"])
    eh, vh, xh = EO[:, 0, POSE], VO[:, 0, POSE], ex["obs"][:, 0, POSE]
    # distractor f for fc_cat: another task's goal (fixed derangement)
    der = lambda n: torch.roll(torch.arange(n, device=dev), 1)
    ef, vf, xf = eg[der(E)], vg[der(NV)], xg[der(N)]
    F = None
    if args.fail_pool:
        SP = torch.load(args.fail_pool, map_location=dev, weights_only=False)
        P, fl = SP["pool"], ~SP["pool"]["succ"].bool()
        F = P["peg"][fl]
        pg = SP["EO"][torch.arange(len(SP["EO"])), final_idx(SP["EA"])][:, PEG]
        CF = (pg[P["task"]] - P["peg"])[fl]  # each failure's own error vector c
        print(f"failure pool: {len(F)} failed final poses from {args.fail_pool}", flush=True)
    draw = lambda n: F[torch.randint(0, len(F), (n,), device=dev)]
    drawc = lambda n: args.fail_scale * CF[torch.randint(0, len(CF), (n,), device=dev)]

    out = json.load(open(f"{run.path}/results.json")) if os.path.exists(f"{run.path}/results.json") else {}
    out["expert"] = ex["succ"].float().mean().item()
    for v in args.variants.split(","):
        if v in out:
            continue
        torch.manual_seed(args.seed * 1000 + VARIANTS.index(v))
        if v in ("fc_fail", "fc_failrel"):
            assert F is not None, f"{v} needs --fail-pool"
            ef, vf, xf = (draw(E), draw(NV), draw(N)) if v == "fc_fail" else (eg - drawc(E), vg - drawc(NV), xg - drawc(N))
        L = {"hole_cat": 1, "g_cat": 1, "fc_cat": 2, "fc_fail": 2, "fc_failrel": 2}.get(v, 0)
        tf = TF(A, L, args.d, args.layers, args.heads, 0.0, True).to(dev)
        Q, C, Vd = inputs(v, EO, eh, eg, ef, E)
        tf.set_norm(Q, C)
        vq, vc, vv = inputs(v, VO, vh, vg, vf, NV)
        opt = torch.optim.AdamW(tf.parameters(), lr=args.lr)
        lam = lambda it: min(1.0, (it + 1) / args.warmup) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * it / args.steps)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
        sel = lambda x, i: None if x is None else x[i]

        def mse(i, q, c, vd, lab, al):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                p = tf(q[i], sel(c, i), sel(vd, i)).float()
            m = al[i].float()
            return (((p - lab[i]) ** 2).mean(-1) * m).sum() / m.sum().clamp_min(1.0)

        curve = []
        tf.train()
        for it in range(args.steps):
            i = torch.randint(0, E, (args.batch_size,), device=dev)
            if v in ("fc_fail", "fc_failrel"):  # redraw the failure for every sample
                C = C.clone()
                C[i] = inputs(v, EO[i, :1], eh[i], eg[i], draw(len(i)) if v == "fc_fail" else eg[i] - drawc(len(i)), len(i))[1]
            loss = mse(i, Q, C, Vd, EL, EA)
            opt.zero_grad(), loss.backward()
            nn.utils.clip_grad_norm_(tf.parameters(), 1.0)
            opt.step(), sched.step()
            if it % 250 == 0 or it == args.steps - 1:
                tf.eval()
                with torch.no_grad():
                    vl = mse(torch.arange(NV, device=dev), vq, vc, vv, VL, VA).item()
                tf.train()
                curve.append((it, loss.item(), vl))
        tf.eval()
        torch.save(tf.state_dict(), run.ckpt(f"{v}.pt"))
        r = rollout(venv, expert, Wrap(tf, v, xh, xg, xf), args, eval_goals, args.eval_seed, "student", dict(s=None, a=None, y=None, valid=None), "normal", mask_pose=False)
        out[v] = dict(succ=r["succ"].float().mean().item(), loss=curve[-1][1], val=curve[-1][2], curve=curve)
        if v.startswith("fc_"):  # diagnostic: same model, f = g and c = 0 (does it read f as the goal?)
            r0 = rollout(venv, expert, Wrap(tf, v, xh, xg, xg), args, eval_goals, args.eval_seed, "student", dict(s=None, a=None, y=None, valid=None), "normal", mask_pose=False)
            out[v]["succ_f_eq_g"] = r0["succ"].float().mean().item()
            print(f"  {v} with f = g, c = 0: succ {out[v]['succ_f_eq_g']:.3f}", flush=True)
        run.save_json("results.json", out)
        print(f"d{args.demos} s{args.seed} {v}: succ {out[v]['succ']:.3f} train {curve[-1][1]:.5f} val {curve[-1][2]:.5f}", flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
