"""Offline BC vs ICL on PegInsertionSideBias-v1, whole workspace (docs/claude/experiments/2026-11-04-offline-bc-vs-icl.md).

Data (per seed): K goals over the whole workspace. Optimal = successful expert trajectory (true goal). Suboptimal = expert acting on a random wrong goal
(failures only) + per-step feedback computed with the TRUE hole pose. Nested prefixes of size N use the first N goals.
BC: trained on the N optimal trajectories. ICL: same N optimal trajectories as queries; w.p. 0.5 no context, w.p. 0.5 the paired suboptimal
trajectory + feedback is the context (target is still the optimal trajectory). The student always sees the true goal, no masking, no online data.
Eval: 512 fixed goals; pass@1 = attempt 1 (no ctx); pass@2 = attempt 1 or 2 (ICL: attempt 2 conditioned on attempt 1 + feedback; BC: independent reset).
"""
import json
import time
from dataclasses import dataclass

import gymnasium as gym
import torch
import torch.nn as nn
import tyro

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from icl_checker import Policy, ctx_index, empty_ctx, make_ctx, rollout, S_DIM, Y_DIM
from icl_tasks import checker_sample
from ppo import Agent
from run_logging import LOG_ROOT, RunDir


@dataclass
class Args:
    expert_checkpoint: str = str(LOG_ROOT / "runs/peg-expert-r1/ckpt_3576.pt")
    exp_name: str = "offline"
    out_dir: str = ""
    sizes: str = "10,20,50,100,200"
    num_envs: int = 512
    num_steps: int = 100
    steps: int = 3000
    batch_size: int = 64
    lr: float = 3e-4
    ctx_stride: int = 5
    ctx_prob: float = 0.5
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    seed: int = 1
    eval_seed: int = 12345
    ctx_feedback: str = "feedback"
    """feedback: per-step (hole - head, head, cos/sin yaw) | expert_action: the expert's action (true goal) at each context state"""
    log_every: int = 0
    """log train loss and held-out val loss (expert-action MSE on the eval goals, no context) every this many steps"""
    split: str = "wide_corner"
    """only used for its xy / yaw range; goals come from region 'all'"""


def main():
    args = tyro.cli(Args)
    device = torch.device("cuda")
    sizes = [int(x) for x in args.sizes.split(",")]
    K = max(sizes)
    run = RunDir(args.out_dir or str(LOG_ROOT / "runs" / f"offline-s{args.seed}"), args)
    torch.manual_seed(args.seed)
    N = args.num_envs
    env = gym.make("PegInsertionSideBias-v1", num_envs=N, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    venv = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)
    A = venv.single_action_space.shape[0]
    assert args.ctx_feedback != "expert_action" or A == Y_DIM, (A, Y_DIM)

    class _Space:
        single_observation_space = type("S", (), {"shape": (43,)})
        single_action_space = venv.single_action_space

    expert = Agent(_Space).to(device)
    expert.load_state_dict(torch.load(args.expert_checkpoint, map_location=device))
    expert.eval()
    Tc = len(ctx_index(args.num_steps, args.ctx_stride, device))
    full = lambda n, gen=None: checker_sample(n, "all", device, gen, args.split)

    # ---- collect: env i keeps its peg geometry, so a pair shares an env index; redo until the expert solves the goal and fails on the wrong goal
    OPT, CTX, nsucc, ntot, sub_lens = {k: [] for k in ("obs", "lab", "alive")}, {k: [] for k in ("s", "a", "y", "valid")}, 0, 0, []
    have, b = 0, 0
    while have < K:
        sd = args.seed * 100000 + b * 20
        goals, fake = full(N), full(N)
        ex = rollout(venv, expert, None, args, goals, sd, "expert")
        wr = rollout(venv, expert, None, args, goals, sd + 1, "wrong_expert", fake=fake)
        for r in range(8):  # resample wrong goals that accidentally succeed
            bad = wr["succ"]
            if not bad.any():
                break
            fake = torch.where(bad[:, None], full(N), fake)
            w2 = rollout(venv, expert, None, args, goals, sd + 2 + r, "wrong_expert", fake=fake)
            for k in wr:
                wr[k] = torch.where(bad.view(-1, *[1] * (wr[k].dim() - 1)), w2[k], wr[k])
        ok = (ex["succ"] & ~wr["succ"]).nonzero()[:, 0]
        for k in OPT:
            OPT[k].append(ex[k][ok])
        if args.ctx_feedback == "expert_action":
            wr["y"] = wr["lab"]
        for k, v in make_ctx(wr, args.ctx_stride).items():
            CTX[k].append(v[ok])
        have += len(ok)
        nsucc += ex["succ"].sum().item()
        ntot += N
        b += 1
    print(f"expert solves {nsucc / ntot:.3f} of goals; collected {have} pairs in {b} batches", flush=True)
    opt = {k: torch.cat(v)[:K] for k, v in OPT.items()}
    ctx = {k: torch.cat(v)[:K] for k, v in CTX.items()}
    sub_len = 100.0

    # ---- fixed eval goals
    g = torch.Generator(device=device).manual_seed(args.eval_seed)
    eval_goals = full(N, g)
    ref = rollout(venv, expert, None, args, eval_goals, args.eval_seed, "expert")["succ"].float().mean().item()
    print(f"expert on eval goals: {ref:.3f}", flush=True)

    vd = rollout(venv, expert, None, args, eval_goals, args.eval_seed + 7, "expert")

    def val_loss(policy):
        policy.eval()
        with torch.no_grad():
            c = empty_ctx(N, Tc, A, device)
            memory, pad = policy.encode(c["s"], c["a"], c["y"], c["valid"])
            mm = vd["alive"].float()
            v = (((policy.decode(vd["obs"], memory, pad) - vd["lab"]) ** 2).mean(-1) * mm).sum() / mm.sum()
        policy.train()
        return v.item()

    def evaluate(policy, icl, ctx_mode="normal"):
        policy.eval()
        e1 = rollout(venv, expert, policy, args, eval_goals, args.eval_seed, "student", empty_ctx(N, Tc, A, device))
        if args.ctx_feedback == "expert_action":
            e1["y"] = e1["lab"]
        e2 = rollout(venv, expert, policy, args, eval_goals, args.eval_seed + 1, "student", make_ctx(e1, args.ctx_stride),
                     ctx_mode if icl else "no_ctx")
        return dict(pass1=e1["succ"].float().mean().item(), pass2=(e1["succ"] | e2["succ"]).float().mean().item(), attempt2=e2["succ"].float().mean().item())

    results = dict(expert_ref=ref, suboptimal_len=sub_len, runs={})
    for n in sizes:
        for method in ("bc", "icl"):
            icl = method == "icl"
            torch.manual_seed(args.seed * 100 + n)
            policy = Policy(A, Tc, args).to(device)
            q, lab, m = opt["obs"][:n], opt["lab"][:n], opt["alive"][:n].float()
            policy.set_norm(q, ctx["y"][:n] if icl else torch.zeros(1, Y_DIM, device=device))
            o = torch.optim.Adam(policy.parameters(), lr=args.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=args.steps, eta_min=args.lr * 0.02)
            z = empty_ctx(args.batch_size, Tc, A, device)
            t0 = time.time()
            for _ in range(args.steps):
                idx = torch.randint(0, n, (args.batch_size,), device=device)
                if icl:
                    use = torch.rand(args.batch_size, device=device) < args.ctx_prob
                    cs, ca, cy, cv = ctx["s"][idx], ctx["a"][idx], ctx["y"][idx], ctx["valid"][idx] & use[:, None]
                else:
                    cs, ca, cy, cv = z["s"], z["a"], z["y"], z["valid"]
                memory, pad = policy.encode(cs, ca, cy, cv)
                loss = (((policy.decode(q[idx], memory, pad) - lab[idx]) ** 2).mean(-1) * m[idx]).sum() / m[idx].sum().clamp_min(1.0)
                o.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                o.step()
                sched.step()
                if args.log_every and (_ + 1) % args.log_every == 0:
                    print(f"  [{method} N={n}] step {_+1} train {loss.item():.5f} val {val_loss(policy):.5f}", flush=True)
            res = dict(loss=loss.item(), val_loss=val_loss(policy), eval=evaluate(policy, icl))
            if icl:
                res["controls"] = {cm: evaluate(policy, icl, cm) for cm in ("no_ctx", "shuf_y", "no_y")}
            results["runs"][f"{method}-{n}"] = res
            print(f"N={n:4d} {method:3s} loss {res['loss']:.4f} pass@1 {res['eval']['pass1']:.3f} pass@2 {res['eval']['pass2']:.3f} ({time.time()-t0:.0f}s)"
                  + (f" | ctrl pass@2 " + " ".join(f"{k} {v['pass2']:.3f}" for k, v in res["controls"].items()) if icl else ""), flush=True)
            run.save_json("results.json", results)


if __name__ == "__main__":
    main()
