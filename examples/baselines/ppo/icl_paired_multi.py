"""Paired offline ICL, hidden goal, v3: ONE small expert dataset (n_init x n_goal ~ 500 trajectories), MANY shared suboptimal trajectories.
 * Configuration = (initial robot + peg state [reset seed, env index]) x (hole pose). Each of the n_init initial states gets n_goal different hole poses.
 * Expert trajectories: expert on (init i, goal g) -> success kept (this is the whole expert dataset; the hidden-goal BC and the visible-goal BC are trained on exactly these).
 * Suboptimal trajectories: for every init i, n_fail runs of the visible-goal BC in an env whose hole is at a RANDOM pose f (BC sees f). They are
   shared by all goals of init i: the pair (expert (i, g), suboptimal (i, k)) is built on the fly with the feedback RELABELED with the true goal g:
   y_t = [g_pos - peg_head_t, peg_head_t, cos/sin g_yaw, |tcp - peg center|] from the stored states (goal part of the context states is masked).
 * Student: same transformer ICL policy / BC-matching regulariser as icl_paired.py; evaluated like it (attempt 1 hidden-goal, attempt 2 with its own attempt as context).
Run: python icl_paired_multi.py [--n-init 100 --n-goal 5 --n-fail 20 --fail-sizes 1,5,20]"""
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
    n_init: int = 100
    n_goal: int = 5
    n_fail: int = 20
    fail_sizes: str = "1,5,20"
    """nested numbers of suboptimal trajectories per initial state used for training"""
    strides: str = "5"
    out_dir: str = str(LOG_ROOT / "runs/icl-paired-multi")


def train_bc(bq, blab, balive, Tc, A, args, dev):
    bc = Policy(A, Tc, args).to(dev)
    bc.set_norm(bq, torch.zeros(1, ic.Y_DIM, device=dev))
    opt = torch.optim.Adam(bc.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.bc_steps, eta_min=args.lr * 0.02)
    z = empty_ctx(args.batch_size, Tc, A, dev)
    for _ in range(args.bc_steps):
        idx = torch.randint(0, len(bq), (args.batch_size,), device=dev)
        memory, pad = bc.encode(z["s"], z["a"], z["y"], z["valid"])
        loss = (((bc.decode(bq[idx], memory, pad) - blab[idx]) ** 2).mean(-1) * balive[idx]).sum() / balive[idx].sum().clamp_min(1.0)
        opt.zero_grad(), loss.backward()
        nn.utils.clip_grad_norm_(bc.parameters(), 1.0)
        opt.step(), sched.step()
    bc.eval()
    for p in bc.parameters():
        p.requires_grad_(False)
    return bc, loss.item()


def main():
    args = tyro.cli(Args)
    dev = torch.device("cuda")
    run = RunDir(args.out_dir, args)
    torch.manual_seed(args.seed)
    N, T, I, G, KF = args.num_envs, args.num_steps, args.n_init, args.n_goal, args.n_fail
    assert I <= N
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
    full = lambda n, gen=None: checker_sample(n, "all", dev, gen, args.split)
    allm = lambda k: torch.ones(k, dtype=torch.bool, device=dev)
    Tc5 = len(ctx_index(T, 5, dev))
    sc = lambda e: e["succ"].float().mean().item()
    SD = args.seed * 100000  # ONE reset seed: env index i = initial state i; goals vary per batch
    init0 = None
    init_diff = 0.0

    def check_init(o):
        nonlocal init0, init_diff
        o = o[:I, 0, :35]  # qpos, qvel, tcp, peg pose, peg half size: everything but the hole
        init0 = o if init0 is None else init0
        init_diff = max(init_diff, (o - init0).abs().max().item())

    # ---- 1. expert dataset: n_goal goals for each of n_init initial states
    EX = {k: [] for k in ("obs", "lab", "alive", "init", "goal")}
    for g_i in range(G):
        goals = full(N)
        r = rollout(venv, expert, None, args, goals, SD, "expert")
        check_init(r["obs"])
        ok = r["succ"][:I].nonzero()[:, 0]
        EX["obs"].append(r["obs"][ok]), EX["lab"].append(r["lab"][ok]), EX["alive"].append(r["alive"][ok])
        EX["init"].append(ok), EX["goal"].append(goals[ok])
    EX = {k: torch.cat(v) for k, v in EX.items()}
    E = len(EX["init"])
    print(f"expert dataset: {E} trajectories = {I} initial states x {G} goals (kept successes); max initial-state diff across batches {init_diff:.2e}", flush=True)

    # ---- 2. BC policies on exactly these expert trajectories
    bq_vis, balive = EX["obs"], EX["alive"].float()
    bc_vis, lv = train_bc(bq_vis, EX["lab"], balive, Tc5, A, args, dev)
    bc_hid, lh = train_bc(apply_mask(bq_vis, allm(E)), EX["lab"], balive, Tc5, A, args, dev)
    torch.save(bc_hid.state_dict(), run.ckpt("bc-hidden.pt")), torch.save(bc_vis.state_dict(), run.ckpt("bc-visible.pt"))
    print(f"BC trained on {E} expert trajectories: visible loss {lv:.5f}, hidden loss {lh:.5f}", flush=True)

    # ---- 3. suboptimal trajectories: visible BC on a RANDOM hole pose, n_fail per initial state (same init, different random hole)
    F = {k: [] for k in ("obs", "act", "alive", "succ")}
    for k in range(KF):
        f = full(N)
        r = rollout(venv, expert, bc_vis, args, f, SD, "student", empty_ctx(N, Tc5, A, dev), mask_pose=False)
        check_init(r["obs"])
        F["obs"].append(r["obs"][:I]), F["act"].append(r["act"][:I]), F["alive"].append(r["alive"][:I]), F["succ"].append(r["succ"][:I])
    F = {k: torch.stack(v, 1) for k, v in F.items()}  # (I, KF, ...)
    print(f"{I * KF} suboptimal trajectories ({KF} per initial state); visible BC solves its own random hole in {F['succ'].float().mean():.3f} of them; "
          f"max initial-state diff over all batches {init_diff:.2e}", flush=True)
    torch.save({**{f"ex_{k}": v.cpu() for k, v in EX.items()}, **{f"fail_{k}": v.cpu() for k, v in F.items()}}, f"{run.path}/data.pt")

    cidx = ctx_index(T, 5, dev)
    qs, mask_all = apply_mask(EX["obs"], allm(E)), allm(args.batch_size)
    gpose = EX["obs"][:, 0, 35:42]

    def batch(idx, kuse):
        """context + queries for expert trajectories idx, each paired with a random suboptimal trajectory of the same initial state, feedback relabeled with the expert's goal."""
        B = len(idx)
        ii, kk = EX["init"][idx], torch.randint(0, kuse, (B,), device=dev)
        fo, fa, fv = F["obs"][ii, kk], F["act"][ii, kk], F["alive"][ii, kk]
        fo2 = fo.clone()
        fo2[..., 35:42] = gpose[idx][:, None]
        y = ic.step_feedback(fo2)
        ctx = dict(s=apply_mask(fo, allm(B))[:, cidx], a=fa[:, cidx], y=y[:, cidx], valid=fv[:, cidx])
        return ctx, apply_mask(fo, allm(B)), fv.float()

    # ---- eval configurations (fresh initial states and goals)
    g = torch.Generator(device=dev).manual_seed(args.eval_seed)
    eval_goals = full(N, g)
    ref = sc(rollout(venv, expert, None, args, eval_goals, args.eval_seed, "expert"))
    z0 = empty_ctx(N, Tc5, A, dev)
    hb = rollout(venv, expert, bc_hid, args, eval_goals, args.eval_seed, "student", z0, mask_pose=True)
    vb = rollout(venv, expert, bc_vis, args, eval_goals, args.eval_seed, "student", z0, mask_pose=False)
    results = dict(n_expert=E, init_diff=init_diff, expert_ref=ref, hidden_bc=sc(hb), visible_bc=sc(vb), runs={})
    print(f"eval: expert {ref:.3f}, visible-goal BC {sc(vb):.3f}, hidden-goal BC {sc(hb):.3f} (all on the {E} expert trajectories)", flush=True)

    # ---- 4. student per number of suboptimal trajectories per init
    bc_sd = {k: v for k, v in bc_hid.state_dict().items() if k != "pos.weight" and not k.startswith("y_")}
    for kuse in [int(x) for x in args.fail_sizes.split(",")]:
        torch.manual_seed(args.seed * 100 + kuse)
        pol = Policy(A, Tc5, args).to(dev)
        pol.load_state_dict(bc_sd, strict=False)
        c_all, _, _ = batch(torch.arange(E, device=dev), kuse)
        yv = c_all["y"][c_all["valid"]]
        pol.y_mean.copy_(yv.mean(0)), pol.y_std.copy_(yv.std(0).clamp_min(1e-3))
        opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.02)
        zb = empty_ctx(args.batch_size, Tc5, A, dev)
        t0 = time.time()
        for it in range(args.steps):
            idx = torch.randint(0, E, (args.batch_size,), device=dev)
            ctx, fq, mf = batch(idx, kuse)
            ms, lab_, q = EX["alive"][idx].float(), EX["lab"][idx], qs[idx]
            memory, pad = pol.encode(ctx["s"], ctx["a"], ctx["y"], ctx["valid"])
            l_ctx = (((pol.decode(q, memory, pad) - lab_) ** 2).mean(-1) * ms).sum() / ms.sum().clamp_min(1.0)
            with torch.no_grad():
                mb, pb = bc_hid.encode(zb["s"], zb["a"], zb["y"], zb["valid"])
                tf, ts = bc_hid.decode(fq, mb, pb), bc_hid.decode(q, mb, pb)
            m0, p0 = pol.encode(zb["s"], zb["a"], zb["y"], zb["valid"])
            l_f = (((pol.decode(fq, m0, p0) - tf) ** 2).mean(-1) * mf).sum() / mf.sum().clamp_min(1.0)
            l_s = (((pol.decode(q, m0, p0) - ts) ** 2).mean(-1) * ms).sum() / ms.sum().clamp_min(1.0)
            loss = l_ctx + args.reg_weight * 0.5 * (l_f + l_s)
            opt.zero_grad(), loss.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step(), sched.step()
            if (it + 1) % args.log_every == 0:
                print(f"  [K={kuse}] step {it + 1} ctx {l_ctx.item():.5f} reg {(0.5 * (l_f + l_s)).item():.5f}", flush=True)
        pol.eval()
        e1 = rollout(venv, expert, pol, args, eval_goals, args.eval_seed, "student", z0, mask_pose=True)
        c2 = make_ctx(e1, 5)
        out = dict(pass1=sc(e1), loss_ctx=l_ctx.item())
        for name in ("normal", "no_ctx", "shuf"):
            if name == "shuf":
                cc = {k: torch.roll(x, 1, 0) for k, x in c2.items()}
                e2 = rollout(venv, expert, pol, args, eval_goals, args.eval_seed, "student", cc, mask_pose=True)
            else:
                e2 = rollout(venv, expert, pol, args, eval_goals, args.eval_seed, "student", c2, name, mask_pose=True)
            out["pass2" if name == "normal" else f"{name}_pass2"] = (e1["succ"] | e2["succ"]).float().mean().item()
            if name == "normal":
                out["attempt2"] = sc(e2)
        results["runs"][f"K{kuse}"] = out
        torch.save(pol.state_dict(), run.ckpt(f"icl-K{kuse}.pt"))
        print(f"K={kuse:3d} failures/init ({time.time() - t0:.0f}s): pass@1 {out['pass1']:.3f} pass@2 {out['pass2']:.3f} attempt2 {out['attempt2']:.3f} | "
              f"no_ctx {out['no_ctx_pass2']:.3f} shuf_ctx {out['shuf_pass2']:.3f} | train ctx loss {out['loss_ctx']:.5f}", flush=True)
        run.save_json("results.json", results)


if __name__ == "__main__":
    main()
