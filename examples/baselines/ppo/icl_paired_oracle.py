"""Oracle sanity check for icl_paired.py: reuse its 500 saved pairs, but the context is ONE token that holds only the true hole pose
(hole position, cos/sin yaw; states / actions zero), queries masked. If the pipeline is sound this should approach the visible-goal BC
trained on the same 500 success trajectories. Run: python icl_paired_oracle.py [src_run [out_run [sizes]]]; defaults runs/icl-paired-s1, runs/icl-paired-oracle, 100,200,500."""
import sys
import time
import gymnasium as gym
import torch
import torch.nn as nn

import icl_paired as ip  # patches icl_checker to the 9-d feedback
import icl_checker as ic
from icl_checker import Policy, apply_mask, empty_ctx, rollout
from icl_tasks import checker_sample
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo import Agent
from run_logging import LOG_ROOT, RunDir

args = ip.Args()
dev = torch.device("cuda")
src = LOG_ROOT / (sys.argv[1] if len(sys.argv) > 1 else "runs/icl-paired-s1")
P = {k: v.to(dev) for k, v in torch.load(src / "data.pt").items()}
run = RunDir(str(LOG_ROOT / (sys.argv[2] if len(sys.argv) > 2 else "runs/icl-paired-oracle")), args)
N, K = args.num_envs, len(P["sd"])
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


def oracle_y(y0):
    """y0 (B, 9) feedback at step 0 -> (B, 9) holding only [hole pos (3), 0 (3), cos, sin, 0]."""
    out = torch.zeros_like(y0)
    out[:, :3] = y0[:, :3] + y0[:, 3:6]
    out[:, 6:8] = y0[:, 6:8]
    return out


def oracle_ctx(y0):
    B = len(y0)
    return dict(s=apply_mask(torch.zeros(B, 1, ic.S_DIM, device=dev), allm(B)), a=torch.zeros(B, 1, A, device=dev),
                y=oracle_y(y0)[:, None], valid=torch.ones(B, 1, dtype=torch.bool, device=dev))


def train(n, oracle):
    torch.manual_seed(100 + n)
    pol = Policy(A, 1, args).to(dev)
    q = apply_mask(P["opt_obs"], allm(K)) if oracle else P["opt_obs"]
    lab, m = P["opt_lab"], P["opt_alive"].float()
    ctx = oracle_ctx(P["f_y"][:, 0])
    pol.set_norm(q[:n], ctx["y"][:n] if oracle else torch.zeros(1, ic.Y_DIM, device=dev))
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.02)
    z = empty_ctx(args.batch_size, 1, A, dev)
    for _ in range(args.steps):
        idx = torch.randint(0, n, (args.batch_size,), device=dev)
        c = {k: v[idx] for k, v in ctx.items()} if oracle else z
        mem, pad = pol.encode(c["s"], c["a"], c["y"], c["valid"])
        loss = (((pol.decode(q[idx], mem, pad) - lab[idx]) ** 2).mean(-1) * m[idx]).sum() / m[idx].sum().clamp_min(1.0)
        opt.zero_grad(), loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step(), sched.step()
    pol.eval()
    return pol, loss.item()


g = torch.Generator(device=dev).manual_seed(args.eval_seed)
eval_goals = checker_sample(N, "all", dev, g, args.split)
ex = rollout(venv, expert, None, args, eval_goals, args.eval_seed, "expert")
ectx = oracle_ctx(ex["y"][:, 0])
res = dict(expert=ex["succ"].float().mean().item())
print(f"expert {res['expert']:.3f}", flush=True)
for n in [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "100,200,500").split(",")]:
    t0 = time.time()
    po, lo = train(n, True)
    pv, lv = train(n, False)
    sc = lambda e: e["succ"].float().mean().item()
    shuf = {k: torch.roll(v, 1, 0) for k, v in ectx.items()}
    r = dict(oracle_ctx=sc(rollout(venv, expert, po, args, eval_goals, args.eval_seed, "student", ectx, mask_pose=True)),
             oracle_shuffled=sc(rollout(venv, expert, po, args, eval_goals, args.eval_seed, "student", shuf, mask_pose=True)),
             oracle_no_ctx=sc(rollout(venv, expert, po, args, eval_goals, args.eval_seed, "student", {**ectx, "valid": torch.zeros_like(ectx["valid"])}, mask_pose=True)),
             visible_bc=sc(rollout(venv, expert, pv, args, eval_goals, args.eval_seed, "student", empty_ctx(N, 1, A, dev), mask_pose=False)),
             loss_oracle=lo, loss_visible=lv)
    res[f"N{n}"] = r
    print(f"N={n:3d} ({time.time() - t0:.0f}s): oracle-context student {r['oracle_ctx']:.3f} | shuffled ctx {r['oracle_shuffled']:.3f} | no ctx {r['oracle_no_ctx']:.3f} | visible-goal BC {r['visible_bc']:.3f} | train loss oracle {lo:.5f} visible {lv:.5f}", flush=True)
    run.save_json("results.json", res)
