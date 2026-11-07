"""Paired offline ICL on PegInsertionSideBias-v1 with a HIDDEN goal (plan: docs/claude/ideas/2026-11-06-paired-offline-icl-plan-v2.md).

Every student input has the hole pose masked (zeros + flag): queries, context states, training and eval, BC and ICL. The true hole pose only
reaches the student through the per-step feedback of the previous attempt.

1. Hidden-goal BC: trained from scratch on --bc-demos expert demos (masked inputs, empty context, same Policy class).
2. Paired data: a configuration = (reset seed, env index, hole pose); both rollouts reset with the same seed and goals, so robot start, peg pose
   and geometry, and hole pose are identical. Success = expert (true goal). Failure = the hidden-goal BC (masked, no context). Keep pairs where
   the expert succeeds and BC fails.
3. ICL student (initialised from the BC weights). Loss = context loss (query = success-trajectory states, context = the paired failure,
   target = expert action on the true goal) + reg-weight * regulariser (empty context, query = states of the failure AND success trajectories,
   target = the frozen BC's action), so that the student without context reproduces the BC policy and its attempt-1 failures.
4. Eval on fresh configurations: pass@1 = attempt 1 (no context), pass@2 = attempt 1 or 2 (attempt 2 conditioned on attempt 1 + feedback, SAME
   reset seed and goal), controls no_ctx / shuf_y / no_y, and BC-match diagnostics for attempt 1.
Feedback (9-d, per step, true hole pose): [hole - peg head (3), peg head (3), cos/sin hole yaw, |tcp - peg center|].
Context sweep: --strides 5,10,100 (every 5 frames: 20 tokens, every 10: 10 tokens, 100 = last frame only).
"""
import time
from dataclasses import dataclass

import gymnasium as gym
import torch
import torch.nn as nn
import tyro

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import icl_checker as ic
from icl_checker import Policy, apply_mask, ctx_index, empty_ctx, make_ctx, rollout
from icl_tasks import checker_sample
from ppo import Agent
from run_logging import LOG_ROOT, RunDir

# 9-d feedback: patch the (process-local) feedback / dimension of icl_checker so its rollout, Policy and empty_ctx use it.
ic.Y_DIM = 9
_step_feedback8 = ic.step_feedback


def step_feedback9(obs):
    return torch.cat([_step_feedback8(obs), (obs[..., 18:21] - obs[..., 25:28]).norm(dim=-1, keepdim=True)], -1)


ic.step_feedback = step_feedback9


@dataclass
class Args:
    expert_checkpoint: str = str(LOG_ROOT / "runs/peg-expert-r1/ckpt_3576.pt")
    out_dir: str = ""
    bc_demos: int = 500
    n_pairs: int = 500
    sizes: str = "100,200,500"
    """nested numbers of pairs used to train the student (<= n_pairs)"""
    strides: str = "5,10,100"
    """context frame strides; 100 = last frame only (num_steps = 100)"""
    num_envs: int = 512
    num_steps: int = 100
    bc_steps: int = 6000
    steps: int = 6000
    batch_size: int = 64
    lr: float = 3e-4
    reg_weight: float = 1.0
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    seed: int = 1
    eval_seed: int = 12345
    split: str = "wide_corner"
    """only used for its xy / yaw range; goals come from region 'all'"""
    log_every: int = 1000


def main():
    args = tyro.cli(Args)
    device = torch.device("cuda")
    sizes, strides = [int(x) for x in args.sizes.split(",")], [int(x) for x in args.strides.split(",")]
    assert max(sizes) <= args.n_pairs
    run = RunDir(args.out_dir or str(LOG_ROOT / "runs" / f"icl-paired-s{args.seed}"), args)
    torch.manual_seed(args.seed)
    N, T = args.num_envs, args.num_steps
    env = gym.make("PegInsertionSideBias-v1", num_envs=N, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    venv = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)
    A = venv.single_action_space.shape[0]

    class _Space:
        single_observation_space = type("S", (), {"shape": (43,)})
        single_action_space = venv.single_action_space

    expert = Agent(_Space).to(device)
    expert.load_state_dict(torch.load(args.expert_checkpoint, map_location=device))
    expert.eval()
    full = lambda n, gen=None: checker_sample(n, "all", device, gen, args.split)
    allm = lambda k: torch.ones(k, dtype=torch.bool, device=device)
    Tc5 = len(ctx_index(T, 5, device))
    D = ic.S_DIM

    def pass_rate(e):
        return e["succ"].float().mean().item()

    # ---- 1. hidden-goal BC on expert demos
    obs_l, lab_l, alive_l, have, b = [], [], [], 0, 0
    while have < args.bc_demos:
        sd = args.seed * 100000 + b * 20
        r = rollout(venv, expert, None, args, full(N), sd, "expert")
        ok = r["succ"].nonzero()[:, 0]
        obs_l.append(r["obs"][ok]), lab_l.append(r["lab"][ok]), alive_l.append(r["alive"][ok])
        have += len(ok)
        b += 1
    bq = apply_mask(torch.cat(obs_l)[: args.bc_demos], allm(args.bc_demos))
    blab, balive = torch.cat(lab_l)[: args.bc_demos], torch.cat(alive_l)[: args.bc_demos].float()
    bc = Policy(A, Tc5, args).to(device)
    bc.set_norm(bq, torch.zeros(1, ic.Y_DIM, device=device))
    opt = torch.optim.Adam(bc.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.bc_steps, eta_min=args.lr * 0.02)
    z = empty_ctx(args.batch_size, Tc5, A, device)
    for _ in range(args.bc_steps):
        idx = torch.randint(0, len(bq), (args.batch_size,), device=device)
        memory, pad = bc.encode(z["s"], z["a"], z["y"], z["valid"])
        loss = (((bc.decode(bq[idx], memory, pad) - blab[idx]) ** 2).mean(-1) * balive[idx]).sum() / balive[idx].sum().clamp_min(1.0)
        opt.zero_grad(), loss.backward()
        nn.utils.clip_grad_norm_(bc.parameters(), 1.0)
        opt.step(), sched.step()
    bc.eval()
    for p in bc.parameters():
        p.requires_grad_(False)
    torch.save(bc.state_dict(), run.ckpt("bc-hidden.pt"))
    print(f"hidden-goal BC trained on {len(bq)} demos, final loss {loss.item():.5f}", flush=True)

    def bc_attempt(goals, seed):
        return rollout(venv, expert, bc, args, goals, seed, "student", empty_ctx(N, Tc5, A, device), mask_pose=True)

    # ---- 2. paired data: same reset seed + goals for the expert and the BC failure
    P = {k: [] for k in ("opt_obs", "opt_lab", "opt_alive", "f_obs", "f_act", "f_lab", "f_y", "f_alive", "sd", "env", "goal")}
    have, b, init_diff = 0, 0, 0.0
    ncfg = nexp = nbc = 0
    while have < args.n_pairs:
        sd = args.seed * 100000 + 50000 + b * 20
        goals = full(N)
        ex = rollout(venv, expert, None, args, goals, sd, "expert")
        fl = bc_attempt(goals, sd)
        init_diff = max(init_diff, (ex["obs"][:, 0] - fl["obs"][:, 0]).abs().max().item())
        ok = (ex["succ"] & ~fl["succ"]).nonzero()[:, 0]
        for k, v in (("opt_obs", ex["obs"]), ("opt_lab", ex["lab"]), ("opt_alive", ex["alive"]), ("f_obs", fl["obs"]), ("f_act", fl["act"]),
                     ("f_lab", fl["lab"]), ("f_y", fl["y"]), ("f_alive", fl["alive"])):
            P[k].append(v[ok])
        P["sd"].append(torch.full((len(ok),), sd, device=device)), P["env"].append(ok), P["goal"].append(goals[ok])
        have += len(ok)
        ncfg += N
        nexp += ex["succ"].sum().item()
        nbc += fl["succ"].sum().item()
        b += 1
    P = {k: torch.cat(v)[: args.n_pairs] for k, v in P.items()}
    print(f"collected {len(P['sd'])} pairs from {ncfg} configs (expert solves {nexp / ncfg:.3f}, hidden BC solves {nbc / ncfg:.3f}); "
          f"max |initial obs diff| between the two rollouts of a pair: {init_diff:.2e}", flush=True)
    torch.save({k: v.cpu() for k, v in P.items()}, f"{run.path}/data.pt")

    # masked training tensors
    K = args.n_pairs
    qs, lab, ms = apply_mask(P["opt_obs"], allm(K)), P["opt_lab"], P["opt_alive"].float()
    fq, mf = apply_mask(P["f_obs"], allm(K)), P["f_alive"].float()
    with torch.no_grad():
        mem0, pad0 = bc.encode(*(lambda c: (c["s"], c["a"], c["y"], c["valid"]))(empty_ctx(K, Tc5, A, device)))
        bc_f, bc_s = bc.decode(fq, mem0, pad0), bc.decode(qs, mem0, pad0)

    # ---- eval configurations
    g = torch.Generator(device=device).manual_seed(args.eval_seed)
    eval_goals = full(N, g)
    ref = pass_rate(rollout(venv, expert, None, args, eval_goals, args.eval_seed, "expert"))
    b1 = bc_attempt(eval_goals, args.eval_seed)
    b2 = bc_attempt(eval_goals, args.eval_seed)  # identical reset again
    b3 = bc_attempt(eval_goals, args.eval_seed + 1)  # new reset
    results = dict(expert_ref=ref, init_diff=init_diff, n_cfg=ncfg, expert_solve=nexp / ncfg, bc_solve=nbc / ncfg,
                   bc=dict(pass1=pass_rate(b1), pass2_same_reset=(b1["succ"] | b2["succ"]).float().mean().item(),
                           pass2_new_reset=(b1["succ"] | b3["succ"]).float().mean().item()), runs={})
    print(f"expert {ref:.3f}; hidden BC pass@1 {results['bc']['pass1']:.3f} pass@2 same reset {results['bc']['pass2_same_reset']:.3f} new reset {results['bc']['pass2_new_reset']:.3f}", flush=True)
    with torch.no_grad():
        mem_e, pad_e = bc.encode(*(lambda c: (c["s"], c["a"], c["y"], c["valid"]))(empty_ctx(N, Tc5, A, device)))
        bc_eval_act = bc.decode(apply_mask(b1["obs"], allm(N)), mem_e, pad_e)

    def evaluate(pol, stride, Tc):
        pol.eval()
        z0 = empty_ctx(N, Tc, A, device)
        e1 = rollout(venv, expert, pol, args, eval_goals, args.eval_seed, "student", z0, mask_pose=True)
        ctx = make_ctx(e1, stride)
        out = dict(pass1=pass_rate(e1))
        for name in ("normal", "no_ctx", "shuf_y", "no_y"):
            e2 = rollout(venv, expert, pol, args, eval_goals, args.eval_seed, "student", ctx, name, mask_pose=True)
            out["pass2" if name == "normal" else f"{name}_pass2"] = (e1["succ"] | e2["succ"]).float().mean().item()
            if name == "normal":
                out["attempt2"] = pass_rate(e2)
        # match with the BC policy on attempt 1: same reset, so near-identical trajectories are expected when they agree
        both = (e1["alive"] & b1["alive"]).float()
        dh = (e1["y"][:, -1, 3:6] - b1["y"][:, -1, 3:6]).norm(dim=-1)
        with torch.no_grad():
            mem, pad = pol.encode(z0["s"], z0["a"], z0["y"], z0["valid"])
            on_bc = (((pol.decode(apply_mask(b1["obs"], allm(N)), mem, pad) - bc_eval_act) ** 2).mean(-1) * b1["alive"].float()).sum() / b1["alive"].float().sum()
        out.update(match_action_mse_on_bc_states=on_bc.item(), match_final_head_dist_median=dh.median().item(),
                   match_final_head_within_1cm=(dh < 0.01).float().mean().item(),
                   match_traj_action_mse=(((e1["act"] - b1["act"]) ** 2).mean(-1) * both).sum().item() / both.sum().item())
        return out

    # ---- 3. ICL student per (size, stride)
    bc_sd = {k: v for k, v in bc.state_dict().items() if k != "pos.weight" and not k.startswith("y_")}
    for stride in strides:
        Tc = len(ctx_index(T, stride, device))
        ctx = make_ctx(dict(obs=P["f_obs"], act=P["f_act"], y=P["f_y"], alive=P["f_alive"]), stride)
        ctx["s"] = apply_mask(ctx["s"], allm(K))
        for n in sizes:
            torch.manual_seed(args.seed * 100 + n + stride)
            pol = Policy(A, Tc, args).to(device)
            pol.load_state_dict(bc_sd, strict=False)  # s_mean/s_std, weights from BC; pos embedding and y norm are new
            yv = P["f_y"][:n][P["f_alive"][:n]]
            pol.y_mean.copy_(yv.mean(0)), pol.y_std.copy_(yv.std(0).clamp_min(1e-3))
            opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.02)
            zb = empty_ctx(args.batch_size, Tc, A, device)
            t0 = time.time()
            for it in range(args.steps + 1):
                idx = torch.randint(0, n, (args.batch_size,), device=device)
                memory, pad = pol.encode(ctx["s"][idx], ctx["a"][idx], ctx["y"][idx], ctx["valid"][idx])
                l_ctx = (((pol.decode(qs[idx], memory, pad) - lab[idx]) ** 2).mean(-1) * ms[idx]).sum() / ms[idx].sum().clamp_min(1.0)
                m0, p0 = pol.encode(zb["s"], zb["a"], zb["y"], zb["valid"])
                l_f = (((pol.decode(fq[idx], m0, p0) - bc_f[idx]) ** 2).mean(-1) * mf[idx]).sum() / mf[idx].sum().clamp_min(1.0)
                l_s = (((pol.decode(qs[idx], m0, p0) - bc_s[idx]) ** 2).mean(-1) * ms[idx]).sum() / ms[idx].sum().clamp_min(1.0)
                l_reg = 0.5 * (l_f + l_s)
                if it == 0:
                    print(f"  [stride {stride} N={n}] init: ctx loss {l_ctx.item():.5f}, BC-match reg loss {l_reg.item():.2e} (should be ~0)", flush=True)
                if it == args.steps:
                    break
                loss = l_ctx + args.reg_weight * l_reg
                opt.zero_grad(), loss.backward()
                nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                opt.step(), sched.step()
                if args.log_every and (it + 1) % args.log_every == 0:
                    print(f"  [stride {stride} N={n}] step {it + 1} ctx {l_ctx.item():.5f} reg {l_reg.item():.5f}", flush=True)
            res = evaluate(pol, stride, Tc)
            res.update(loss_ctx=l_ctx.item(), loss_reg=l_reg.item())
            torch.save(pol.state_dict(), run.ckpt(f"icl-N{n}-stride{stride}.pt"))
            results["runs"][f"icl-N{n}-stride{stride}"] = res
            print(f"stride {stride:3d} N={n:4d} ({time.time() - t0:.0f}s): pass@1 {res['pass1']:.3f} pass@2 {res['pass2']:.3f} attempt2 {res['attempt2']:.3f} | "
                  f"no_ctx {res['no_ctx_pass2']:.3f} shuf_y {res['shuf_y_pass2']:.3f} no_y {res['no_y_pass2']:.3f} | match: act-mse(BC states) {res['match_action_mse_on_bc_states']:.2e} "
                  f"final-head median {res['match_final_head_dist_median']:.3f} m, within 1cm {res['match_final_head_within_1cm']:.2f}", flush=True)
            run.save_json("results.json", results)


if __name__ == "__main__":
    main()
