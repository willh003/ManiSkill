"""Transformer offline-ICL ablation on peg insertion (hidden goal). Companion of icl_final_state.py (MLP) with the same data pipeline.
 - T0 `bc`: transformer visible-goal BC (obs token only) on the 500 expert demos.
 - ICL students are an encoder over [obs token, context pose tokens]; context built from a suboptimal BC rollout paired with a random expert demo
   (training), or from the student's previous attempt (eval, sliding window of 1 attempt). Poses are 7-d peg poses (xyz + quat).
   f = final peg pose of the failed attempt, g = expert final peg pose on that config, c = g - f, hist = peg pose every 10 steps (steps 0..90, clipped to the final step).
 kinds:  fcfc        [f, c, f+c]                                  (T1)
         fcfc_hist   [f, c, f+c, hist x10]                        (A)
         hist_c      [hist x10, f, c]       (goal not given directly)  (B)
         hist_c_noisy hist_c with c scaled by s~U(.9,1.1) (fresh per sample / per attempt)  (C)
 Eval: recursive attempts k=1..K; attempt k (k>1) sees only attempt k-1. Reports success, pass@k, peg-goal distance (attempt-k and best-so-far)."""
import math
import time
from dataclasses import dataclass
import gymnasium as gym
import torch
import torch.nn as nn
import tyro

from icl_checker import apply_mask, rollout
from icl_tasks import checker_sample
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo import Agent
from run_logging import LOG_ROOT, RunDir

PEG = slice(25, 32)
NH = 10  # history frames: steps 0,10,...,90
LAYOUT = dict(hole=1, goal=1, fc=2, fcfc=3, fcfc_hist=3 + NH, hist_c=NH + 2, hist_c_noisy=NH + 2,
              fg=2, rg=2, gf=2, cg=2, gfc=3, cfg=3)


@dataclass
class Args:
    expert_checkpoint: str = str(LOG_ROOT / "runs/peg-expert-r1/ckpt_3576.pt")
    out_dir: str = str(LOG_ROOT / "runs/icl-tf")
    bc_demos: int = 500
    ns: str = "100,1000,10000"
    kinds: str = "fcfc,fcfc_hist,hist_c,hist_c_noisy"
    num_envs: int = 512
    num_steps: int = 100
    steps: int = 10000
    warmup: int = 500
    eval_every: int = 5000
    K: int = 8  # max attempts at final eval
    batch_size: int = 128
    lr: float = 3e-4
    d: int = 128
    layers: int = 2
    heads: int = 4
    noise_lo: float = 0.9
    noise_hi: float = 1.1
    seed: int = 1
    eval_seed: int = 12345
    split: str = "wide_corner"
    skip_bc: bool = False  # skip transformer BC (T0)
    dropout: float = 0.0
    wd: float = 0.0
    fp32: bool = False
    no_slot: bool = False  # debug: zero, freeze the per-slot embeddings
    no_cnorm: bool = False  # debug: raw (unnormalised) context
    bc_only: bool = False  # train only the visible-goal transformer BC, saving a checkpoint + (val loss, success) at every eval_every steps, then exit
    cat: bool = False  # debug: concatenate the (flattened, normalised, validity-gated) context to the obs token instead of separate context tokens


class TF(nn.Module):
    """action = head(Encoder([obs token, ctx tokens])[obs]). L = number of context pose tokens (0 -> BC). forward(s (B,T,43), c (B,L,7) raw, valid (B,))."""

    def __init__(self, A, L, d, layers, heads, dropout=0.0, cat=False):
        super().__init__()
        self.L, self.A, self.cat = L, A, cat and L > 0
        self.register_buffer("s_mean", torch.zeros(43)), self.register_buffer("s_std", torch.ones(43))
        self.register_buffer("c_mean", torch.zeros(max(L, 1), 7)), self.register_buffer("c_std", torch.ones(max(L, 1), 7))
        self.obs_proj = nn.Linear(43 + (7 * L if self.cat else 0), d)
        self.obs_emb = nn.Parameter(torch.zeros(d))
        if L and not self.cat:
            self.tok_proj = nn.Linear(7, d)
            self.slot_emb = nn.Parameter(torch.randn(L, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, A))

    def set_norm(self, s, c=None):
        self.s_mean.copy_(s.reshape(-1, 43).mean(0)), self.s_std.copy_(s.reshape(-1, 43).std(0).clamp_min(1e-3))
        self.s_mean[42], self.s_std[42] = 0.0, 1.0
        if c is not None:
            self.c_mean.copy_(c.mean(0)), self.c_std.copy_(c.std(0).clamp_min(1e-3))

    def forward(self, s, c=None, valid=None):
        x = (s - self.s_mean) / self.s_std
        B, T = x.shape[:2]
        if self.cat:
            cn = ((c - self.c_mean) / self.c_std).clamp(-5, 5).reshape(B, 1, -1) * valid[:, None, None].float()
            x = torch.cat([x, cn.expand(B, T, -1)], -1)
        o = (self.obs_proj(x) + self.obs_emb)[:, :, None]  # (B,T,1,d)
        if self.L and not self.cat:
            cn = ((c - self.c_mean) / self.c_std).clamp(-5, 5)
            ct = self.tok_proj(cn) + self.slot_emb  # (B,L,d)
            seq = torch.cat([o, ct[:, None].expand(B, T, self.L, -1)], 2)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), (~valid)[:, None].expand(B, self.L)], 1)
            pad = pad[:, None].expand(B, T, 1 + self.L).reshape(B * T, 1 + self.L)
            h = self.enc(seq.reshape(B * T, 1 + self.L, -1), src_key_padding_mask=pad)
        else:
            h = self.enc(o.reshape(B * T, 1, -1))
        return self.head(h[:, 0]).reshape(B, T, self.A)

    # adapter for icl_checker.rollout: ctx y is (B,1,L*7), validity (B,1)
    def encode(self, cs, ca, cy, cv):
        if not self.L:
            return (None, None), None
        return (cy[:, 0].reshape(-1, self.L, 7), cv[:, 0]), None

    def decode(self, x, memory, pad):
        c, v = memory
        return self(x, c, v)


def final_idx(alive):
    return alive.long().sum(1) - 1


def hist_of(obs, fi):
    """obs (B,T,43), fi (B,) -> peg pose at steps 0,10,..,90 clipped to the final step: (B,NH,7)."""
    idx = torch.minimum(torch.arange(NH, device=obs.device)[None] * 10, fi[:, None])
    return torch.gather(obs[:, :, PEG], 1, idx[..., None].expand(-1, -1, 7))


def mk(kind, hist, fp, g, s):
    c = (g - fp) * s[:, None]
    if kind in ("hole", "goal"):  # oracle references: the goal vector alone (g is the exact hole pose for `hole`)
        return g[:, None]
    if kind == "fc":  # diagnostic: [f, c] only (no f+c, no history)
        return torch.stack([fp, c], 1)
    if kind == "fg":
        return torch.stack([fp, g], 1)
    if kind == "rg":  # f of another sample (uncorrelated distractor)
        return torch.stack([torch.roll(fp, 1, 0), g], 1)
    if kind == "gf":
        return torch.stack([g, fp], 1)
    if kind == "cg":
        return torch.stack([c, g], 1)
    if kind == "gfc":
        return torch.stack([g, fp, c], 1)
    if kind == "cfg":
        return torch.stack([c, fp, g], 1)
    if kind == "fcfc":
        return torch.stack([fp, c, fp + c], 1)
    if kind == "fcfc_hist":
        return torch.cat([torch.stack([fp, c, fp + c], 1), hist], 1)
    return torch.cat([hist, fp[:, None], c[:, None]], 1)  # hist_c, hist_c_noisy


def quat_angle(q1, q2):
    return 2 * torch.acos((q1 * q2).sum(-1).abs().clamp(max=1.0))


def main():
    args = tyro.cli(Args)
    torch.backends.cuda.matmul.allow_tf32 = True
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
    one = lambda n: torch.ones(n, device=dev)

    def dummy(n, L):
        return dict(s=torch.zeros(n, 1, 43, device=dev), a=torch.zeros(n, 1, A, device=dev), y=torch.zeros(n, 1, max(L, 1) * 7, device=dev), valid=torch.zeros(n, 1, dtype=torch.bool, device=dev))

    def collect(n_need, seed0):
        ob, lb, al, have, b = [], [], [], 0, 0
        while have < n_need:
            r = rollout(venv, expert, None, args, full(N), seed0 + b * 20, "expert")
            ok = r["succ"].nonzero()[:, 0]
            ob.append(r["obs"][ok]), lb.append(r["lab"][ok]), al.append(r["alive"][ok])
            have += len(ok)
            b += 1
        return (torch.cat(x)[:n_need] for x in (ob, lb, al))

    # ---- 1. expert demos (train + held-out val)
    EO, EL, EA = collect(args.bc_demos, args.seed * 100000)
    E = len(EO)
    Ealive = EA.float()
    e_fi = final_idx(EA)
    e_peg = EO[torch.arange(E), e_fi][:, PEG]
    E_hist = hist_of(EO, e_fi)
    NV = 200
    VO, VL, VA = collect(NV, args.seed * 100000 + 30000)
    v_fi = final_idx(VA)
    v_peg = VO[torch.arange(NV), v_fi][:, PEG]
    VQ, EQ = apply_mask(VO, allm(NV)), apply_mask(EO, allm(E))

    g_ = torch.Generator(device=dev).manual_seed(args.eval_seed)
    eval_goals = full(N, g_)
    ex = rollout(venv, expert, None, args, eval_goals, args.eval_seed, "expert")
    ex_peg = ex["obs"][torch.arange(N), final_idx(ex["alive"])][:, PEG]
    results = dict(expert=sc(ex), runs={})

    def fit(pol, q, ctxf, steps_, curve, val, hook):
        opt = torch.optim.AdamW(pol.parameters(), lr=args.lr, weight_decay=args.wd)
        lam = lambda it: min(1.0, (it + 1) / args.warmup) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * it / steps_)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
        pol.train()
        for it in range(steps_):
            e_idx, c, v = ctxf()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp32):
                pred = pol(q[e_idx], c, v).float()
            m = Ealive[e_idx]
            loss = (((pred - EL[e_idx]) ** 2).mean(-1) * m).sum() / m.sum().clamp_min(1.0)
            opt.zero_grad(), loss.backward()
            gn = nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step(), sched.step()
            if it % 50 == 0:
                curve.append((it, loss.item(), gn.item(), val(pol)))
            if args.eval_every > 0 and (it + 1) % args.eval_every == 0:
                hook(pol, it + 1)
                pol.train()
        pol.eval()
        return loss.item()

    # ---- T0: transformer BC
    if not args.skip_bc:
        bc = TF(A, 0, args.d, args.layers, args.heads, args.dropout).to(dev)
        bc.set_norm(EO)
        curves, trace = dict(bc=[]), dict(bc=[])

        def val_bc(pol):
            pol.eval()
            with torch.no_grad():
                m = VA.float()
                r = ((((pol(VO) - VL) ** 2).mean(-1) * m).sum() / m.sum()).item()
            pol.train()
            return r

        def bc_hook(pol, it):
            pol.eval()
            s_ = sc(rollout(venv, expert, pol, args, eval_goals, args.eval_seed, "student", dummy(N, 0), mask_pose=False))
            v_ = val_bc(pol)
            trace["bc"].append((it, s_, v_))
            run.save_json("trace_bc.json", trace)
            if args.bc_only:
                torch.save(pol.state_dict(), run.ckpt(f"tf-bc-it{it:05d}.pt"))
            print(f"  [tf bc] step {it}: success {s_:.3f} val {v_:.5f}", flush=True)

        t0 = time.time()
        bl = fit(bc, EO, lambda: (torch.randint(0, E, (args.batch_size,), device=dev), None, None), args.steps, curves["bc"], val_bc, bc_hook)
        torch.save(bc.state_dict(), run.ckpt("tf-bc.pt"))
        results["tf_bc"] = dict(success=sc(rollout(venv, expert, bc, args, eval_goals, args.eval_seed, "student", dummy(N, 0), mask_pose=False)), loss=bl, val=val_bc(bc))
        run.save_json("curves_bc.json", curves), run.save_json("results.json", results)
        print(f"transformer BC: success {results['tf_bc']['success']:.3f} loss {bl:.5f} val {results['tf_bc']['val']:.5f} ({time.time() - t0:.0f}s); expert {results['expert']:.3f}", flush=True)

    if args.bc_only:
        return

    # ---- 2. suboptimal rollouts from a visible-goal BC. The rollout source is the MLP BC of the earlier experiments (so data matches the MLP runs), trained here.
    from icl_final_state import MLP
    mbc = MLP(A, 0, 512).to(dev)
    mbc.set_norm(EO)
    opt = torch.optim.Adam(mbc.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=3e-4 * 0.02)
    for it in range(args.steps):
        i = torch.randint(0, E, (128,), device=dev)
        m = Ealive[i]
        loss = (((mbc(EO[i], None, None) - EL[i]) ** 2).mean(-1) * m).sum() / m.sum()
        opt.zero_grad(), loss.backward(), nn.utils.clip_grad_norm_(mbc.parameters(), 1.0), opt.step(), sch.step()
    mbc.eval()
    results["mlp_bc"] = sc(rollout(venv, expert, mbc, args, eval_goals, args.eval_seed, "student", dummy(N, 0), mask_pose=False))
    print(f"MLP BC (rollout source) success {results['mlp_bc']:.3f}", flush=True)
    ns = [int(x) for x in args.ns.split(",")]
    F_hist, F_peg, F_succ, have, b = [], [], [], 0, 0
    while have < max(ns) + 2000:
        r = rollout(venv, expert, mbc, args, full(N), args.seed * 100000 + 60000 + b * 20, "student", dummy(N, 0), mask_pose=False)
        fi = final_idx(r["alive"])
        F_hist.append(hist_of(r["obs"], fi)), F_peg.append(r["obs"][torch.arange(N), fi][:, PEG]), F_succ.append(r["succ"])
        have += N
        b += 1
    F_hist, F_peg, F_succ = (torch.cat(x)[: max(ns) + 2000] for x in (F_hist, F_peg, F_succ))
    print(f"{len(F_peg)} BC rollouts; BC solves {F_succ.float().mean():.3f}", flush=True)
    torch.save(dict(F_hist=F_hist.cpu(), F_peg=F_peg.cpu(), F_succ=F_succ.cpu()), f"{run.path}/rollouts.pt")
    results["rollout_bc_success"] = F_succ.float().mean().item()

    VF = torch.arange(max(ns), max(ns) + 2000, device=dev)
    ve = torch.arange(NV, device=dev).repeat(5)
    vj = VF[torch.randint(0, 2000, (len(ve),), device=dev)]
    rs = lambda n, noisy: (args.noise_lo + (args.noise_hi - args.noise_lo) * torch.rand(n, device=dev)) if noisy else one(n)

    # ---- eval: recursive attempts
    def attempts(pol, kind, K, noisy, ctx_mode="normal", shuffle=False, x_g=None):
        L = LAYOUT[kind]
        out, r = [], None
        for k in range(1, K + 1):
            if k == 1:
                ctx = dummy(N, L)
                r = rollout(venv, expert, pol, args, eval_goals, args.eval_seed, "student", ctx, "normal", mask_pose=True)
            else:
                fi = final_idx(r["alive"])
                fp = r["obs"][torch.arange(N), fi][:, PEG]
                c = mk(kind, hist_of(r["obs"], fi), fp, ex_peg if x_g is None else x_g, rs(N, noisy))
                if shuffle:
                    c = torch.roll(c, 1, 0)
                ctx = dict(s=torch.zeros(N, 1, 43, device=dev), a=torch.zeros(N, 1, A, device=dev), y=c.reshape(N, 1, -1), valid=torch.ones(N, 1, dtype=torch.bool, device=dev))
                r = rollout(venv, expert, pol, args, eval_goals, args.eval_seed, "student", ctx, ctx_mode, mask_pose=True)
            fi = final_idx(r["alive"])
            fpk = r["obs"][torch.arange(N), fi][:, PEG]
            out.append(dict(succ=r["succ"], dist=(fpk[:, :3] - ex_peg[:, :3]).norm(dim=-1), ang=quat_angle(fpk[:, 3:], ex_peg[:, 3:])))
        return out

    def summarize(out):
        succ = torch.stack([o["succ"] for o in out], 1)  # (N,K)
        dist = torch.stack([o["dist"] for o in out], 1)
        ang = torch.stack([o["ang"] for o in out], 1)
        return dict(
            succ=succ.float().mean(0).tolist(),
            pass_k=succ.cummax(1).values.float().mean(0).tolist(),
            dist=dist.mean(0).tolist(), dist_med=dist.median(0).values.tolist(),
            dist_best=dist.cummin(1).values.mean(0).tolist(), dist_best_med=dist.cummin(1).values.median(0).values.tolist(),
            ang=ang.mean(0).tolist(), ang_best=ang.cummin(1).values.mean(0).tolist())

    # ---- 3. ICL students
    kinds = args.kinds.split(",")
    trace, curves = {}, {}
    e_hole, v_hole, ex_hole = EO[:, 0, 35:42], VO[:, 0, 35:42], ex["obs"][:, 0, 35:42]
    for kind in kinds:
        e_g, v_g, x_g = (e_hole, v_hole, ex_hole) if kind == "hole" else (e_peg, v_peg, ex_peg)
        L, noisy = LAYOUT[kind], kind == "hist_c_noisy"
        for n in ns:
            torch.manual_seed(args.seed * 100 + n)
            key = f"{kind}-N{n}"
            Es = torch.randint(0, E, (5000,), device=dev)
            Ej = torch.randint(0, E, (5000,), device=dev)
            Cn = mk(kind, E_hist[Ej], e_peg[Ej], e_g[Es], one(5000))  # N-independent norm stats from the expert demos
            pol = TF(A, L, args.d, args.layers, args.heads, args.dropout, args.cat).to(dev)
            pol.set_norm(EQ)
            pol.c_mean.copy_(Cn.mean(0)), pol.c_std.copy_(Cn.std(0).clamp_min(1e-3))
            if args.no_cnorm:
                pol.c_mean.zero_(), pol.c_std.fill_(1.0)
            if args.no_slot and L and not args.cat:
                pol.slot_emb.data.zero_(), pol.slot_emb.requires_grad_(False)

            def sample():
                j = torch.randint(0, n, (args.batch_size,), device=dev)
                e = torch.randint(0, E, (args.batch_size,), device=dev)
                return e, mk(kind, F_hist[j], F_peg[j], e_g[e], rs(args.batch_size, noisy)), torch.ones(args.batch_size, dtype=torch.bool, device=dev)

            def val_icl(p):
                p.eval()
                with torch.no_grad():
                    c = mk(kind, F_hist[vj], F_peg[vj], v_g[ve], rs(len(ve), noisy))
                    tot = den = 0.0
                    for i in range(0, len(ve), 100):
                        sl = slice(i, i + 100)
                        m = VA[ve[sl]].float()
                        pr = p(VQ[ve[sl]], c[sl], torch.ones(len(m), dtype=torch.bool, device=dev))
                        tot += (((pr - VL[ve[sl]]) ** 2).mean(-1) * m).sum().item()
                        den += m.sum().item()
                    r = tot / den
                p.train()
                return r

            curves[key], trace[key] = [], []

            def hook(p, it):
                p.eval()
                s = summarize(attempts(p, kind, 2, noisy, x_g=x_g))
                trace[key].append((it, s["succ"][0], s["pass_k"][1], s["succ"][1], val_icl(p)))
                run.save_json("trace.json", trace)
                print(f"  [{key}] step {it}: pass@1 {s['succ'][0]:.3f} pass@2 {s['pass_k'][1]:.3f} att2 {s['succ'][1]:.3f}", flush=True)

            t0 = time.time()
            loss = fit(pol, EQ, sample, args.steps, curves[key], val_icl, hook)
            run.save_json("curves.json", curves)
            torch.save(pol.state_dict(), run.ckpt(f"{key}.pt"))
            res = summarize(attempts(pol, kind, args.K, noisy, x_g=x_g))
            res["loss"], res["val"] = loss, val_icl(pol)
            for name, kw in (("no_ctx", dict(ctx_mode="no_ctx")), ("shuf", dict(shuffle=True))):
                res[f"{name}_pass2"] = summarize(attempts(pol, kind, 2, noisy, x_g=x_g, **kw))["pass_k"][1]
            results["runs"][key] = res
            run.save_json("results.json", results)
            print(f"{key}: pass@k {[round(x, 3) for x in res['pass_k']]} dist_best {[round(x, 3) for x in res['dist_best']]} | no_ctx {res['no_ctx_pass2']:.3f} shuf {res['shuf_pass2']:.3f} | loss {loss:.5f} val {res['val']:.5f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
