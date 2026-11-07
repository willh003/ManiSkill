"""Oracle-context check on the icl_paired_multi dataset: context = ONE token holding only the true goal (hole xyz, cos/sin yaw); queries masked.
Trained on the same expert trajectories (logs/runs/icl-paired-multi/data.pt). Compares with the visible-goal BC. Run: python icl_paired_multi_oracle.py"""
import gymnasium as gym
import torch
import torch.nn as nn

import icl_paired as ip
import icl_checker as ic
from icl_checker import Policy, apply_mask, empty_ctx, rollout
from icl_tasks import checker_sample
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo import Agent
from run_logging import LOG_ROOT, RunDir

args = ip.Args()
dev = torch.device("cuda")
D = torch.load(LOG_ROOT / "runs/icl-paired-multi/data.pt")
obs, lab, alive = D["ex_obs"].to(dev), D["ex_lab"].to(dev), D["ex_alive"].to(dev).float()
E, N = len(obs), args.num_envs
run = RunDir(str(LOG_ROOT / "runs/icl-paired-multi-oracle"), args)
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


def oracle_ctx(y0):
    """y0 (B, 9) step-0 feedback -> one-token context with only [hole pos, 0, cos, sin, 0]."""
    y = torch.zeros_like(y0)
    y[:, :3] = y0[:, :3] + y0[:, 3:6]
    y[:, 6:8] = y0[:, 6:8]
    B = len(y0)
    return dict(s=apply_mask(torch.zeros(B, 1, ic.S_DIM, device=dev), allm(B)), a=torch.zeros(B, 1, A, device=dev), y=y[:, None],
                valid=torch.ones(B, 1, dtype=torch.bool, device=dev))


def train(use_ctx):
    torch.manual_seed(1)
    pol = Policy(A, 1, args).to(dev)
    q = apply_mask(obs, allm(E)) if use_ctx else obs
    ctx = oracle_ctx(ic.step_feedback(obs[:, 0]))
    pol.set_norm(q, ctx["y"] if use_ctx else torch.zeros(1, ic.Y_DIM, device=dev))
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.02)
    z = empty_ctx(args.batch_size, 1, A, dev)
    for _ in range(args.steps):
        idx = torch.randint(0, E, (args.batch_size,), device=dev)
        c = {k: v[idx] for k, v in ctx.items()} if use_ctx else z
        mem, pad = pol.encode(c["s"], c["a"], c["y"], c["valid"])
        loss = (((pol.decode(q[idx], mem, pad) - lab[idx]) ** 2).mean(-1) * alive[idx]).sum() / alive[idx].sum().clamp_min(1.0)
        opt.zero_grad(), loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step(), sched.step()
    pol.eval()
    return pol, loss.item()


g = torch.Generator(device=dev).manual_seed(args.eval_seed)
eval_goals = checker_sample(N, "all", dev, g, args.split)
ex = rollout(venv, expert, None, args, eval_goals, args.eval_seed, "expert")
ectx = oracle_ctx(ex["y"][:, 0])
sc = lambda e: e["succ"].float().mean().item()
po, lo = train(True)
pv, lv = train(False)
shuf = {k: torch.roll(v, 1, 0) for k, v in ectx.items()}
r = dict(expert=sc(ex), oracle_ctx=sc(rollout(venv, expert, po, args, eval_goals, args.eval_seed, "student", ectx, mask_pose=True)),
         oracle_shuffled=sc(rollout(venv, expert, po, args, eval_goals, args.eval_seed, "student", shuf, mask_pose=True)),
         oracle_no_ctx=sc(rollout(venv, expert, po, args, eval_goals, args.eval_seed, "student", {**ectx, "valid": torch.zeros_like(ectx["valid"])}, mask_pose=True)),
         visible_bc=sc(rollout(venv, expert, pv, args, eval_goals, args.eval_seed, "student", empty_ctx(N, 1, A, dev), mask_pose=False)),
         loss_oracle=lo, loss_visible=lv, n_expert=E)
print(r, flush=True)
run.save_json("results.json", r)
