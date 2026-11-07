"""Context-content ablation for icl_paired.py (reuses its saved pairs + hidden-goal BC from logs/runs/icl-paired-s1).
Context = the paired failure trajectory subsampled every 5 frames (20 tokens); variants keep only some of its channels:
  hole        per-step true hole pose only [hole xyz, cos, sin] (the oracle info, repeated per token)
  y           9-d feedback only (hole - head, head, cos, sin, gripper-peg distance), states/actions zeroed
  y_s         feedback + states
  full        feedback + states + executed actions (= icl_paired stride 5)
Same BC init, same BC-matching regularizer, masked queries as icl_paired. Run: python icl_paired_ablate.py --sizes 200,500"""
import time
from dataclasses import dataclass
import gymnasium as gym
import torch
import torch.nn as nn
import tyro

import icl_paired as ip  # patches icl_checker to 9-d feedback
import icl_checker as ic
from icl_checker import Policy, apply_mask, ctx_index, empty_ctx, make_ctx, rollout
from icl_tasks import checker_sample
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo import Agent
from run_logging import LOG_ROOT, RunDir


@dataclass
class Args(ip.Args):
    src: str = str(LOG_ROOT / "runs/icl-paired-s1")
    variants: str = "hole,y,y_s,full"
    stride: int = 5


def transform(ctx, v):
    s, a, y = ctx["s"], ctx["a"], ctx["y"]
    if v == "hole":
        h = torch.zeros_like(y)
        h[..., :3] = y[..., :3] + y[..., 3:6]
        h[..., 6:8] = y[..., 6:8]
        y, s, a = h, torch.zeros_like(s), torch.zeros_like(a)
    elif v == "y":
        s, a = torch.zeros_like(s), torch.zeros_like(a)
    elif v == "y_s":
        a = torch.zeros_like(a)
    return dict(s=s, a=a, y=y, valid=ctx["valid"])


def main():
    args = tyro.cli(Args)
    dev = torch.device("cuda")
    P = {k: v.to(dev) for k, v in torch.load(f"{args.src}/data.pt").items()}
    run = RunDir(args.out_dir or str(LOG_ROOT / "runs/icl-paired-ablate"), args)
    N, T, K = args.num_envs, args.num_steps, len(P["sd"])
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
    allm = lambda k: torch.ones(k, dtype=torch.bool, device=dev)
    Tc = len(ctx_index(T, args.stride, dev))
    bc = Policy(A, Tc, args).to(dev)
    bc_sd = torch.load(f"{args.src}/checkpoints/bc-hidden.pt", map_location=dev)
    bc_sd = {k: v for k, v in bc_sd.items() if k != "pos.weight"}
    bc.load_state_dict(bc_sd, strict=False)
    bc.eval()
    qs, lab, ms = apply_mask(P["opt_obs"], allm(K)), P["opt_lab"], P["opt_alive"].float()
    fq, mf = apply_mask(P["f_obs"], allm(K)), P["f_alive"].float()
    zK = empty_ctx(K, Tc, A, dev)
    with torch.no_grad():
        m0_, p0_ = bc.encode(zK["s"], zK["a"], zK["y"], zK["valid"])
        bc_f, bc_s = bc.decode(fq, m0_, p0_), bc.decode(qs, m0_, p0_)
    base = make_ctx(dict(obs=P["f_obs"], act=P["f_act"], y=P["f_y"], alive=P["f_alive"]), args.stride)
    base["s"] = apply_mask(base["s"], allm(K))

    g = torch.Generator(device=dev).manual_seed(args.eval_seed)
    eval_goals = checker_sample(N, "all", dev, g, args.split)
    sc = lambda e: e["succ"].float().mean().item()
    z0 = empty_ctx(N, Tc, A, dev)
    results = {}
    for v in args.variants.split(","):
        ctx = transform(base, v)
        for n in [int(x) for x in args.sizes.split(",")]:
            torch.manual_seed(args.seed * 100 + n + 7)
            pol = Policy(A, Tc, args).to(dev)
            pol.load_state_dict({k: x for k, x in bc_sd.items() if not k.startswith("y_")}, strict=False)
            yv = ctx["y"][:n][ctx["valid"][:n]]
            pol.y_mean.copy_(yv.mean(0)), pol.y_std.copy_(yv.std(0).clamp_min(1e-3))
            opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.02)
            zb = empty_ctx(args.batch_size, Tc, A, dev)
            t0 = time.time()
            for it in range(args.steps):
                idx = torch.randint(0, n, (args.batch_size,), device=dev)
                mem, pad = pol.encode(ctx["s"][idx], ctx["a"][idx], ctx["y"][idx], ctx["valid"][idx])
                l_ctx = (((pol.decode(qs[idx], mem, pad) - lab[idx]) ** 2).mean(-1) * ms[idx]).sum() / ms[idx].sum().clamp_min(1.0)
                m0, p0 = pol.encode(zb["s"], zb["a"], zb["y"], zb["valid"])
                l_f = (((pol.decode(fq[idx], m0, p0) - bc_f[idx]) ** 2).mean(-1) * mf[idx]).sum() / mf[idx].sum().clamp_min(1.0)
                l_s = (((pol.decode(qs[idx], m0, p0) - bc_s[idx]) ** 2).mean(-1) * ms[idx]).sum() / ms[idx].sum().clamp_min(1.0)
                loss = l_ctx + args.reg_weight * 0.5 * (l_f + l_s)
                opt.zero_grad(), loss.backward()
                nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                opt.step(), sched.step()
            pol.eval()
            e1 = rollout(venv, expert, pol, args, eval_goals, args.eval_seed, "student", z0, mask_pose=True)
            c2 = transform(make_ctx(e1, args.stride), v)
            out = dict(pass1=sc(e1), loss_ctx=l_ctx.item())
            for name in ("normal", "no_ctx", "shuf_y"):
                if name == "shuf_y":  # shuffle the whole context across envs
                    cc = {k: torch.roll(x, 1, 0) for k, x in c2.items()}
                    e2 = rollout(venv, expert, pol, args, eval_goals, args.eval_seed, "student", cc, mask_pose=True)
                else:
                    e2 = rollout(venv, expert, pol, args, eval_goals, args.eval_seed, "student", c2, name, mask_pose=True)
                out["pass2" if name == "normal" else f"{name}_pass2"] = (e1["succ"] | e2["succ"]).float().mean().item()
                if name == "normal":
                    out["attempt2"] = sc(e2)
            # teacher-forced held-out check: loss on queries of the last 100 pairs is not available (n<=K), skip
            results[f"{v}-N{n}"] = out
            torch.save(pol.state_dict(), run.ckpt(f"{v}-N{n}.pt"))
            print(f"{v:5s} N={n:4d} ({time.time() - t0:.0f}s): pass@1 {out['pass1']:.3f} pass@2 {out['pass2']:.3f} attempt2 {out['attempt2']:.3f} | "
                  f"no_ctx {out['no_ctx_pass2']:.3f} shuf_ctx {out['shuf_y_pass2']:.3f} | train ctx loss {out['loss_ctx']:.5f}", flush=True)
            run.save_json("results.json", results)


if __name__ == "__main__":
    main()
