"""On-policy outer loop for the peg hidden-hole ICL transformer (docs/claude/experiments/2026-11-16-icl-onpolicy-loop.md).

 round 0: visible-hole transformer BC on E expert demos, rolled out once on every demo task -> pool of rollouts.
 round r=1..R: train the hole-hidden ICL transformer (concat context pathway) from scratch on the pool; a sample is
     (context built from a pooled rollout, expert demo as target); optionally a fraction p_noctx of samples has no context (default 0, see Args).
   Evaluate k=1..K recursive attempts on 512 fixed eval tasks (attempt 1 blind, attempt k sees attempt k-1 only).
   Then roll the model out K times on every demo task with the same sliding window and append all K rollouts to the pool.
 Context of a rollout with final peg pose f on a task whose expert final peg pose is g (c = g - f, h = peg pose every 10 steps):
   --kind goal [f+c] | fc [f, c] | hist_c [h x10, f, c]
 --pairing matched: context from a rollout on task t, target = t's expert demo.
           random : f/h from a random pooled rollout, c recomputed against the target demo's goal.
 --supervise demo  : targets are the expert demos only (fixed expert budget).
             dagger: half of each batch is the student's own visited states with expert action labels, conditioned on the
                     context that attempt actually had (attempts 2..K only: no-context samples make the model ignore the context).
State is saved after every round (state.pt) and the script resumes from it."""
import math
import os
import time
from dataclasses import dataclass

import gymnasium as gym
import torch
import torch.nn as nn
import tyro

from icl_checker import apply_mask, rollout
from icl_tasks import checker_sample
from icl_transformer import LAYOUT, PEG, TF, final_idx, hist_of, mk, quat_angle
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo import Agent
from run_logging import LOG_ROOT, RunDir


@dataclass
class Args:
    expert_checkpoint: str = str(LOG_ROOT / "runs/peg-expert-r1/ckpt_3576.pt")
    out_dir: str = str(LOG_ROOT / "runs/icl-op")
    demos: int = 100
    val_demos: int = 200
    kind: str = "fc"  # goal | fc | hist_c
    pairing: str = "matched"  # matched | random
    supervise: str = "demo"  # demo | dagger
    rounds: int = 6
    K: int = 8
    p_noctx: float = 0.0  # fraction of samples with no context. Keep 0: at 100 demos 0.125 makes the model fit the demos from the state alone and ignore the context (goal-only pass@8 .08 -> .004)
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
    eval_seed: int = 12345
    split: str = "wide_corner"


def main():
    args = tyro.cli(Args)
    assert args.kind in ("goal", "fc", "hist_c") and args.pairing in ("matched", "random") and args.supervise in ("demo", "dagger")
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    run = RunDir(args.out_dir, args)
    torch.manual_seed(args.seed)
    N, K, R, kind = args.num_envs, args.K, args.rounds, args.kind
    L, matched, dagger = LAYOUT[kind], args.pairing == "matched", args.supervise == "dagger"
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
    ar = torch.arange(N, device=dev)
    ones = lambda n: torch.ones(n, device=dev)
    allm = lambda n: torch.ones(n, dtype=torch.bool, device=dev)
    fin = lambda r: r["obs"][ar, final_idx(r["alive"])][:, PEG]

    def ctx_of(y, valid):
        return dict(s=torch.zeros(N, 1, 43, device=dev), a=torch.zeros(N, 1, A, device=dev), y=y.reshape(N, 1, -1), valid=torch.full((N, 1), valid, dtype=torch.bool, device=dev))

    def collect(n_need, seed0):
        """Expert demos (successes only) + the info to replay each demo's task: batch seed, hole poses, env slots."""
        ob, lb, al, bt, have, b = [], [], [], [], 0, 0
        while have < n_need:
            goals, seed = checker_sample(N, "all", dev, None, args.split), seed0 + b * 20
            r = rollout(venv, expert, None, args, goals, seed, "expert")
            ok = r["succ"].nonzero()[:, 0][: n_need - have]
            ob.append(r["obs"][ok]), lb.append(r["lab"][ok]), al.append(r["alive"][ok])
            bt.append(dict(seed=seed, goals=goals, gpeg=fin(r), slots=ok, idx=torch.arange(have, have + len(ok), device=dev)))
            have += len(ok)
            b += 1
        return torch.cat(ob), torch.cat(lb), torch.cat(al), bt

    sp = f"{run.path}/state.pt"
    if os.path.exists(sp):
        S = torch.load(sp, map_location=dev, weights_only=False)
        print(f"resumed from {sp} at round {S['round']}", flush=True)
    else:
        S = dict(round=0, pool=None, vpool=None, curves={})
        S["EO"], S["EL"], S["EA"], S["EB"] = collect(args.demos, args.seed * 100000)
        S["VO"], S["VL"], S["VA"], S["VB"] = collect(args.val_demos, args.seed * 100000 + 30000)
        S["eval_goals"] = checker_sample(N, "all", dev, torch.Generator(device=dev).manual_seed(args.eval_seed), args.split)
        ex = rollout(venv, expert, None, args, S["eval_goals"], args.eval_seed, "expert")
        S["ex_peg"] = fin(ex)
        S["results"] = dict(expert=ex["succ"].float().mean().item(), rounds={})
    EO, EL, EA, EB, VO, VL, VA, VB = (S[k] for k in ("EO", "EL", "EA", "EB", "VO", "VL", "VA", "VB"))
    eval_goals, ex_peg, results, curves = S["eval_goals"], S["ex_peg"], S["results"], S["curves"]
    E, NV = len(EO), len(VO)
    e_peg, v_peg = EO[torch.arange(E), final_idx(EA)][:, PEG], VO[torch.arange(NV), final_idx(VA)][:, PEG]
    EQ, VQ = apply_mask(EO, allm(E)), apply_mask(VO, allm(NV))

    def save_state():
        torch.save(S, sp + ".tmp")
        os.replace(sp + ".tmp", sp)
        run.save_json("results.json", results), run.save_json("curves.json", curves)

    # ---- pool of rollouts on the demo tasks
    def collect_pool(pool, batches, pol, rnd, bc=False):
        """Roll `pol` out on the demo tasks (K sliding-window attempts; one visible-hole attempt if bc) and append to the pool.
        Returns the pool and the per-attempt success rate on the demo tasks."""
        new, succ = [], []
        base = 0 if pool is None else len(pool["peg"])
        for bt in batches:
            r, prev = None, None
            for k in range(1 if bc else K):
                if bc:
                    r = rollout(venv, expert, pol, args, bt["goals"], bt["seed"], "student", ctx_of(torch.zeros(N, 7, device=dev), False), "normal", mask_pose=False)
                elif k == 0:
                    r = rollout(venv, expert, pol, args, bt["goals"], bt["seed"], "student", ctx_of(torch.zeros(N, L * 7, device=dev), False), "normal", mask_pose=True)
                else:
                    c = mk(kind, hist_of(r["obs"], final_idx(r["alive"])), fin(r), bt["gpeg"], ones(N))
                    r = rollout(venv, expert, pol, args, bt["goals"], bt["seed"], "student", ctx_of(c, True), "normal", mask_pose=True)
                sl, n = bt["slots"], len(bt["slots"])
                d = dict(hist=hist_of(r["obs"], final_idx(r["alive"]))[sl], peg=fin(r)[sl], task=bt["idx"], succ=r["succ"][sl],
                         rnd=torch.full((n,), rnd, device=dev), att=torch.full((n,), k, device=dev))
                if dagger:
                    d.update(obs=apply_mask(r["obs"][sl], allm(n)), lab=r["lab"][sl], alive=r["alive"][sl], q=torch.full((n,), not bc and prev is not None, dtype=torch.bool, device=dev),
                             ctx=torch.full((n,), -1, device=dev) if prev is None else prev)
                prev = base + torch.arange(n, device=dev)
                base += n
                new.append(d)
        succ = [torch.cat([d["succ"] for d in new if d["att"][0] == k]).float().mean().item() for k in range(1 if bc else K)]
        parts = ([pool] if pool is not None else []) + new
        return {key: torch.cat([p[key] for p in parts]) for key in new[0]}, succ

    def batch(pool, Q, Lb, Al, gp, B, p_noctx):
        """-> query obs (B,T,43) hole masked, labels, alive mask, context (B,L,7), context-valid (B,)."""
        j = torch.randint(0, len(pool["peg"]), (B,), device=dev)
        e = pool["task"][j] if matched else torch.randint(0, len(Q), (B,), device=dev)
        c = mk(kind, pool["hist"][j], pool["peg"][j], gp[e], ones(B))
        valid = torch.rand(B, device=dev) >= p_noctx
        q, lab, al = Q[e], Lb[e], Al[e]
        qi = pool["q"].nonzero()[:, 0] if dagger else []
        if len(qi):  # relabelled on-policy states: second half of the batch
            h = B // 2
            i = qi[torch.randint(0, len(qi), (h,), device=dev)]
            cj = pool["ctx"][i].clamp_min(0)
            c2 = mk(kind, pool["hist"][cj], pool["peg"][cj], gp[pool["task"][i]], ones(h))
            q, lab, al = torch.cat([q[: B - h], pool["obs"][i]]), torch.cat([lab[: B - h], pool["lab"][i]]), torch.cat([al[: B - h], pool["alive"][i]])
            c, valid = torch.cat([c[: B - h], c2]), torch.cat([valid[: B - h], pool["ctx"][i] >= 0])
        return q, lab, al.float(), c, valid

    def mse(pol, q, lab, al, c, v):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = pol(q, c, v).float()
        return (((pred - lab) ** 2).mean(-1) * al).sum() / al.sum().clamp_min(1.0)

    def fit(pol, sampler, valset, curve, tag):
        opt = torch.optim.AdamW(pol.parameters(), lr=args.lr)
        lam = lambda it: min(1.0, (it + 1) / args.warmup) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * it / args.steps)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)

        def val():
            pol.eval()
            with torch.no_grad():
                tot = sum(mse(pol, *(x[i : i + 250] for x in valset)).item() * valset[2][i : i + 250].sum().item() for i in range(0, len(valset[0]), 250))
            pol.train()
            return tot / valset[2].sum().item()

        pol.train()
        for it in range(args.steps):
            loss = mse(pol, *sampler())
            opt.zero_grad(), loss.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step(), sched.step()
            if it % 250 == 0 or it == args.steps - 1:
                curve.append((it, loss.item(), val()))
            if it % 2500 == 0:
                print(f"  [{tag}] step {it}: train {curve[-1][1]:.5f} val {curve[-1][2]:.5f}", flush=True)
        pol.eval()
        return curve[-1][1], curve[-1][2]

    # ---- eval: K recursive attempts on the fixed eval tasks
    def attempts(pol, mode="real"):
        """mode real: context = previous attempt | fixed: attempt-1 context reused | randf: another task's attempt-1 f/h with the exact goal."""
        r = rollout(venv, expert, pol, args, eval_goals, args.eval_seed, "student", ctx_of(torch.zeros(N, L * 7, device=dev), False), "normal", mask_pose=True)
        f1, h1, out = fin(r), hist_of(r["obs"], final_idx(r["alive"])), []
        for k in range(K):
            if k:
                if mode == "real":
                    f, h = fin(r), hist_of(r["obs"], final_idx(r["alive"]))
                elif mode == "fixed":
                    f, h = f1, h1
                else:
                    pm = torch.randperm(N, device=dev)
                    f, h = f1[pm], h1[pm]
                r = rollout(venv, expert, pol, args, eval_goals, args.eval_seed, "student", ctx_of(mk(kind, h, f, ex_peg, ones(N)), True), "normal", mask_pose=True)
            fp = fin(r)
            out.append(dict(succ=r["succ"], dist=(fp[:, :3] - ex_peg[:, :3]).norm(dim=-1), ang=quat_angle(fp[:, 3:], ex_peg[:, 3:])))
        succ, dist = torch.stack([o["succ"] for o in out], 1), torch.stack([o["dist"] for o in out], 1)
        return dict(succ=succ.float().mean(0).tolist(), pass_k=succ.cummax(1).values.float().mean(0).tolist(),
                    dist=dist.mean(0).tolist(), dist_best=dist.cummin(1).values.mean(0).tolist())

    r3 = lambda x: [round(v, 3) for v in x]
    # ---- round 0: visible-hole BC, one rollout per demo task
    if S["round"] == 0:
        t0 = time.time()
        bc = TF(A, 0, args.d, args.layers, args.heads).to(dev)
        bc.set_norm(EO)
        curves["bc"] = []
        fit(bc, lambda: (lambda e: (EO[e], EL[e], EA[e].float(), None, None))(torch.randint(0, E, (args.batch_size,), device=dev)),
            (VO, VL, VA.float(), [None] * NV, [None] * NV), curves["bc"], "bc")
        torch.save(bc.state_dict(), run.ckpt("tf-bc.pt"))
        results["bc"] = rollout(venv, expert, bc, args, eval_goals, args.eval_seed, "student", ctx_of(torch.zeros(N, 7, device=dev), False), "normal", mask_pose=False)["succ"].float().mean().item()
        S["pool"], s0 = collect_pool(None, EB, bc, 0, bc=True)
        S["vpool"], _ = collect_pool(None, VB, bc, 0, bc=True)
        results["bc_train_succ"] = s0[0]
        S["round"] = 1
        save_state()
        print(f"round 0: BC success {results['bc']:.3f} (expert {results['expert']:.3f}), on demo tasks {s0[0]:.3f}, pool {len(S['pool']['peg'])} ({time.time() - t0:.0f}s)", flush=True)

    # ---- rounds 1..R: train ICL on the pool, eval, collect on-policy
    for r in range(S["round"], R + 1):
        t0 = time.time()
        torch.manual_seed(args.seed * 1000 + r)
        pool, vpool = S["pool"], S["vpool"]
        pol = TF(A, L, args.d, args.layers, args.heads, 0.0, True).to(dev)
        pol.set_norm(EQ)
        Cn = batch(pool, EQ, EL, EA, e_peg, 5000, 0.0)[3]
        pol.c_mean.copy_(Cn.mean(0)), pol.c_std.copy_(Cn.std(0).clamp_min(1e-3))
        valset = batch(vpool, VQ, VL, VA, v_peg, 1000, 0.0)
        curves[f"r{r}"] = []
        loss, vl = fit(pol, lambda: batch(pool, EQ, EL, EA, e_peg, args.batch_size, args.p_noctx), valset, curves[f"r{r}"], f"r{r}")
        torch.save(pol.state_dict(), run.ckpt(f"round{r}.pt"))
        res = attempts(pol)
        res.update(loss=loss, val=vl, pool=len(pool["peg"]), pool_succ=pool["succ"].float().mean().item())
        if r in (1, R):
            for m in ("fixed", "randf"):
                c = attempts(pol, m)
                res[f"{m}_succ"], res[f"{m}_pass_k"] = c["succ"], c["pass_k"]
        if r < R:
            S["pool"], res["train_succ"] = collect_pool(pool, EB, pol, r)
            S["vpool"], _ = collect_pool(vpool, VB, pol, r)
        results["rounds"][str(r)] = res
        S["round"] = r + 1
        save_state()
        print(f"round {r}: pass@k {r3(res['pass_k'])} perf@k {r3(res['succ'])} | pool {res['pool']} (succ {res['pool_succ']:.3f}) loss {loss:.5f} val {vl:.5f} ({time.time() - t0:.0f}s)", flush=True)
        if "randf_pass_k" in res:
            print(f"   controls: fixed pass@k {r3(res['fixed_pass_k'])} | randf pass@k {r3(res['randf_pass_k'])}", flush=True)
    results["done"] = True
    run.save_json("results.json", results)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
