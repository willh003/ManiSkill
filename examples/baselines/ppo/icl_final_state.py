"""Final-state-correction ICL on peg insertion (hidden goal), MLP policies, no BC regularizer.
 1. 500 expert demos at uniform configs -> visible-goal MLP BC (the baseline; also gives the expert's final peg pose per demo).
 2. Roll the BC out N times from fresh uniform configs; keep each rollout's FINAL state (hole pose masked).
 3. Pair each rollout with a RANDOM expert demo: correction = (expert final peg pose) - (BC final peg pose) (7-d: xyz + quat).
    Training sample: (masked state of the expert demo, context = [masked BC final state (43), correction (7)]) -> expert action, every step.
    The goal is only recoverable as final_peg + correction.
 4. Eval (hidden goal, fresh configs): attempt 1 has no context; attempt 2's context = final state of attempt 1 + (expert final peg pose on the same config - attempt-1 final peg pose).
    BC baseline: visible-goal BC on the 500 demos (pass@1).
Run: python icl_final_state.py --ns 100,300,1000,3000,10000,30000"""
import time
from dataclasses import dataclass
import gymnasium as gym
import torch
import torch.nn as nn
import tyro

import icl_checker as ic
from icl_checker import apply_mask, rollout
from icl_tasks import checker_sample
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo import Agent
from run_logging import LOG_ROOT, RunDir

PEG = slice(25, 32)  # peg pose (xyz + quat) in the 43-d student obs


@dataclass
class Args:
    expert_checkpoint: str = str(LOG_ROOT / "runs/peg-expert-r1/ckpt_3576.pt")
    out_dir: str = str(LOG_ROOT / "runs/icl-final-state")
    bc_demos: int = 500
    ns: str = "100,300,1000,3000,10000,30000"
    num_envs: int = 512
    num_steps: int = 100
    steps: int = 6000
    eval_every: int = 0  # >0: evaluate success every this many grad steps (stored in results['trace'])
    batch_size: int = 128
    lr: float = 3e-4
    hidden: int = 512
    ctx_kind: str = "state"
    """state: context = [masked final state (43), correction (7)]; pose: [final peg pose (7), correction (7)] (no final state);
    goal: [expert final peg pose = goal pose (7)] only (oracle reference, ignores the BC rollouts);
    hole: [true hole pose (7, obs 35:42)] only (same, with the exact hole pose as the goal vector)"""
    resample: bool = True
    """draw a fresh random expert demo for every suboptimal rollout in every batch (False: one fixed random pairing per rollout)"""
    ctx_drop: float = 0.5
    """prob. of an empty context during ICL training (so attempt 1 is in-distribution)"""
    seed: int = 1
    eval_seed: int = 12345
    split: str = "wide_corner"


class MLP(nn.Module):
    """action = MLP([norm state (43), has_ctx, norm ctx (50)]); ctx = [final state (43), correction (7)]. cdim = 0 -> plain BC."""
    def __init__(self, A, cdim, hidden):
        super().__init__()
        self.cdim = cdim
        self.register_buffer("s_mean", torch.zeros(43)), self.register_buffer("s_std", torch.ones(43))
        self.register_buffer("c_mean", torch.zeros(max(cdim, 1))), self.register_buffer("c_std", torch.ones(max(cdim, 1)))
        d_in = 43 + (1 + cdim if cdim else 0)
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, A))

    def set_norm(self, s, c=None):
        self.s_mean.copy_(s.reshape(-1, 43).mean(0)), self.s_std.copy_(s.reshape(-1, 43).std(0).clamp_min(1e-3))
        self.s_mean[42], self.s_std[42] = 0.0, 1.0
        if c is not None:
            self.c_mean.copy_(c.mean(0)), self.c_std.copy_(c.std(0).clamp_min(1e-3))

    def forward(self, s, c=None, valid=None):
        x = (s - self.s_mean) / self.s_std
        if self.cdim:
            B, extra = x.shape[0], [1] * (x.dim() - 2)
            cn = (((c - self.c_mean) / self.c_std) * valid.float()[:, None]).view(B, *extra, self.cdim).expand(*x.shape[:-1], self.cdim)
            vf = valid.float().view(B, *extra, 1).expand(*x.shape[:-1], 1)
            x = torch.cat([x, vf, cn], -1)
        return self.net(x)

    # --- adapter for icl_checker.rollout: the whole context vector is passed as cy (B, 1, cdim), validity as cv (B, 1)
    def encode(self, cs, ca, cy, cv):
        return (cy[:, 0], cv[:, 0]), None

    def decode(self, x, memory, pad):
        c, v = memory
        return self(x, c, v)


def final_idx(alive):
    return alive.long().sum(1) - 1


def main():
    args = tyro.cli(Args)
    dev = torch.device("cuda")
    run = RunDir(args.out_dir, args)
    torch.manual_seed(args.seed)
    N, T = args.num_envs, args.num_steps
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
    sc = lambda e: e["succ"].float().mean().item()
    dummy = lambda n, d: dict(s=torch.zeros(n, 1, 43, device=dev), a=torch.zeros(n, 1, A, device=dev), y=torch.zeros(n, 1, d, device=dev), valid=torch.zeros(n, 1, dtype=torch.bool, device=dev))
    CD = dict(state=50, pose=14, goal=7, hole=7)[args.ctx_kind]
    mk = lambda fs, fp, ep: ep if args.ctx_kind in ("goal", "hole") else (torch.cat([fp, ep - fp], -1) if args.ctx_kind == "pose" else torch.cat([fs, ep - fp], -1))

    # ---- 1. expert demos + visible-goal BC
    ob, lb, al, have, b = [], [], [], 0, 0
    while have < args.bc_demos:
        r = rollout(venv, expert, None, args, full(N), args.seed * 100000 + b * 20, "expert")
        ok = r["succ"].nonzero()[:, 0]
        ob.append(r["obs"][ok]), lb.append(r["lab"][ok]), al.append(r["alive"][ok])
        have += len(ok)
        b += 1
    EO, EL, EA = (torch.cat(x)[: args.bc_demos] for x in (ob, lb, al))
    Ealive = EA.float()
    E = len(EO)
    e_peg = EO[torch.arange(E), final_idx(EA)][:, PEG]  # expert final peg pose per demo
    if args.ctx_kind == "hole":
        e_peg = EO[:, 0, 35:42]
    NV = 200
    vo, vl, va, have, b = [], [], [], 0, 0
    while have < NV:  # held-out expert demos (never used for training), for validation loss
        r = rollout(venv, expert, None, args, full(N), args.seed * 100000 + 30000 + b * 20, "expert")
        ok = r["succ"].nonzero()[:, 0]
        vo.append(r["obs"][ok]), vl.append(r["lab"][ok]), va.append(r["alive"][ok])
        have += len(ok)
        b += 1
    VO, VL, VA = (torch.cat(x)[:NV] for x in (vo, vl, va))
    v_peg = VO[torch.arange(NV), final_idx(VA)][:, PEG] if args.ctx_kind != "hole" else VO[:, 0, 35:42]
    VQ = apply_mask(VO, allm(NV))
    EQ = apply_mask(EO, allm(E))  # hidden-goal queries

    def fit(pol, q, ctxf, steps_, curve=None, val=None, hook=None):
        """ctxf(idx) -> (c, valid) or None. MSE to expert actions on masked/visible queries q (E_,T,43) of the demos given by ctxf's demo index."""
        opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_, eta_min=args.lr * 0.02)
        for it in range(steps_):
            idx, e_idx, c, v = ctxf()
            pred = pol(q[e_idx], c, v)
            m = Ealive[e_idx]
            loss = (((pred - EL[e_idx]) ** 2).mean(-1) * m).sum() / m.sum().clamp_min(1.0)
            opt.zero_grad(), loss.backward()
            gn = nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step(), sched.step()
            if curve is not None and it % 20 == 0:
                curve.append((it, loss.item(), gn.item(), val(pol) if val else float('nan')))
            if hook is not None and args.eval_every > 0 and (it + 1) % args.eval_every == 0:
                hook(pol, it + 1)
        pol.eval()
        return loss.item()

    bc = MLP(A, 0, args.hidden).to(dev)
    bc.set_norm(EO)
    def val_bc(pol):
        with torch.no_grad():
            m = VA.float()
            return ((((pol(VO, None, None) - VL) ** 2).mean(-1) * m).sum() / m.sum()).item()

    curves = dict(bc=[])
    trace = dict(bc=[])

    def bc_hook(pol, it):
        pol.eval()
        sc_ = sc(rollout(venv, expert, pol, args, eval_goals, args.eval_seed, "student", dummy(N, 1), mask_pose=False))
        pol.train()
        trace["bc"].append((it, sc_))
        run.save_json("trace.json", trace)
        print(f"  [bc] step {it}: success {sc_:.3f}", flush=True)

    # eval set (needed by the hooks)
    g = torch.Generator(device=dev).manual_seed(args.eval_seed)
    eval_goals = full(N, g)
    bc_loss = fit(bc, EO, lambda: (None, (i := torch.randint(0, E, (args.batch_size,), device=dev)), None, None), args.steps, curves["bc"], val_bc, bc_hook)
    torch.save(bc.state_dict(), run.ckpt("bc-visible.pt"))
    print(f"visible-goal MLP BC on {E} demos: final loss {bc_loss:.5f}", flush=True)

    # ---- eval set + references
    ex = rollout(venv, expert, None, args, eval_goals, args.eval_seed, "expert")
    ex_peg = ex["obs"][:, 0, 35:42] if args.ctx_kind == "hole" else ex["obs"][torch.arange(N), final_idx(ex["alive"])][:, PEG]
    bc_eval = sc(rollout(venv, expert, bc, args, eval_goals, args.eval_seed, "student", dummy(N, 1), mask_pose=False))
    results = dict(expert=sc(ex), bc_visible=bc_eval, bc_loss=bc_loss, runs={})
    print(f"eval: expert {sc(ex):.3f}, visible-goal BC (500 demos) {bc_eval:.3f}", flush=True)

    # ---- 2. BC rollouts from uniform configs (final states), once for the largest N
    ns = [int(x) for x in args.ns.split(",")]
    F_state, F_peg, F_succ, have, b = [], [], [], 0, 0
    while have < max(ns) + 2000:  # last 2000 are held out for the validation loss
        r = rollout(venv, expert, bc, args, full(N), args.seed * 100000 + 60000 + b * 20, "student", dummy(N, 1), mask_pose=False)
        fi = final_idx(r["alive"])
        fs = r["obs"][torch.arange(N), fi]
        F_state.append(apply_mask(fs, allm(N))), F_peg.append(fs[:, PEG]), F_succ.append(r["succ"])
        have += N
        b += 1
    F_state, F_peg, F_succ = (torch.cat(x)[: max(ns) + 2000] for x in (F_state, F_peg, F_succ))
    print(f"{len(F_state)} BC rollouts from uniform configs; BC solves {F_succ.float().mean():.3f} of them", flush=True)
    torch.save(dict(F_state=F_state.cpu(), F_peg=F_peg.cpu(), F_succ=F_succ.cpu(), e_peg=e_peg.cpu()), f"{run.path}/rollouts.pt")

    # ---- 3. ICL student per N
    VF_ids = torch.arange(max(ns), max(ns) + 2000, device=dev)
    ve = torch.arange(NV, device=dev).repeat(5)  # 1000 fixed (val demo, val rollout) pairs
    vj = VF_ids[torch.randint(0, 2000, (len(ve),), device=dev)]

    def val_icl(pol):
        with torch.no_grad():
            c = mk(F_state[vj], F_peg[vj], v_peg[ve])
            m = VA[ve].float()
            pred = pol(VQ[ve], c, torch.ones(len(ve), dtype=torch.bool, device=dev))
            return ((((pred - VL[ve]) ** 2).mean(-1) * m).sum() / m.sum()).item()

    for n in ns:
        torch.manual_seed(args.seed * 100 + n)
        e_of = torch.randint(0, E, (n,), device=dev)  # random expert demo for every suboptimal rollout
        C = mk(F_state[:n], F_peg[:n], e_peg[e_of])  # (n, CD)
        jn, en = torch.randint(0, n, (5000,), device=dev), torch.randint(0, E, (5000,), device=dev)
        Cn = mk(F_state[jn], F_peg[jn], e_peg[en])  # random pairings, for normalisation
        pol = MLP(A, CD, args.hidden).to(dev)
        pol.set_norm(EQ, Cn)

        def sample():
            j = torch.randint(0, n, (args.batch_size,), device=dev)
            v = torch.rand(args.batch_size, device=dev) >= args.ctx_drop
            if args.resample:
                e = torch.randint(0, E, (args.batch_size,), device=dev)
                return j, e, mk(F_state[j], F_peg[j], e_peg[e]), v
            return j, e_of[j], C[j], v

        curves[f"N{n}"] = []
        trace[f"N{n}"] = []

        def icl_hook(pl, it):
            pl.eval()
            a1 = rollout(venv, expert, pl, args, eval_goals, args.eval_seed, "student", dummy(N, CD), mask_pose=True)
            g1 = a1["obs"][torch.arange(N), final_idx(a1["alive"])]
            c2 = mk(apply_mask(g1, allm(N)), g1[:, PEG], ex_peg)
            cx = dict(s=torch.zeros(N, 1, 43, device=dev), a=torch.zeros(N, 1, A, device=dev), y=c2[:, None], valid=torch.ones(N, 1, dtype=torch.bool, device=dev))
            a2 = rollout(venv, expert, pl, args, eval_goals, args.eval_seed, "student", cx, "normal", mask_pose=True)
            r = (it, sc(a1), (a1["succ"] | a2["succ"]).float().mean().item(), sc(a2), val_icl(pl))
            pl.train()
            trace[f"N{n}"].append(r)
            run.save_json("trace.json", trace)
            print(f"  [N={n}] step {it}: pass@1 {r[1]:.3f} pass@2 {r[2]:.3f} attempt2 {r[3]:.3f} val {r[4]:.5f}", flush=True)

        loss = fit(pol, EQ, sample, args.steps, curves[f"N{n}"], val_icl, icl_hook)
        run.save_json("curves.json", curves)
        e1 = rollout(venv, expert, pol, args, eval_goals, args.eval_seed, "student", dummy(N, CD), mask_pose=True)
        f1 = e1["obs"][torch.arange(N), final_idx(e1["alive"])]
        C2 = mk(apply_mask(f1, allm(N)), f1[:, PEG], ex_peg)
        ctx2 = dict(s=torch.zeros(N, 1, 43, device=dev), a=torch.zeros(N, 1, A, device=dev), y=C2[:, None], valid=torch.ones(N, 1, dtype=torch.bool, device=dev))
        out = dict(pass1=sc(e1), loss=loss)
        for name in ("normal", "no_ctx", "shuf"):
            if name == "shuf":
                c = dict(ctx2, y=torch.roll(ctx2["y"], 1, 0))
                e2 = rollout(venv, expert, pol, args, eval_goals, args.eval_seed, "student", c, mask_pose=True)
            else:
                e2 = rollout(venv, expert, pol, args, eval_goals, args.eval_seed, "student", ctx2, name, mask_pose=True)
            out["pass2" if name == "normal" else f"{name}_pass2"] = (e1["succ"] | e2["succ"]).float().mean().item()
            if name == "normal":
                out["attempt2"] = sc(e2)
        results["runs"][f"N{n}"] = out
        torch.save(pol.state_dict(), run.ckpt(f"icl-N{n}.pt"))
        print(f"N={n:6d}: pass@1 {out['pass1']:.3f} pass@2 {out['pass2']:.3f} attempt2 {out['attempt2']:.3f} | no_ctx {out['no_ctx_pass2']:.3f} shuf {out['shuf_pass2']:.3f} | train loss {loss:.5f}", flush=True)
        run.save_json("results.json", results)


if __name__ == "__main__":
    main()
