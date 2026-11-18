"""ICL (target hidden, right/wrong feedback) vs BC (target given) on PickClutterYCBCond-v1.

Data: for each of `demos` layouts, the target-conditioned PPO expert is rolled out once per condition c in 0..K-1 (same layout,
deterministic robot init). A layout is kept only if all K conditioned rollouts lift their own object, so any (target,
search-history) pair can be composed offline.

ICL sample: target t, a random searcher history over the objects (a random permutation tried in order until t, then t repeated),
its last <= L_CTX attempts as context tokens, and a timestep of the expert(t) rollout as the label. Each context token is the
outcome of that attempt: final object xy (K x 2), lifted flag per object (max lift >= 4cm, K), feedback bit (1 = right object). With no "right"
token the label is uniform over the untried objects, so the policy is a GMM.

BC: same network with no context and the target block visible, trained on the same rollouts (each rollout is a demo of its own
condition).

Eval: `eval_layouts` held-out layouts with random targets; attempts 1..`attempts` each reset the same layout. Reports
perf@k (success of attempt k), pass@k, repeat rate (lifted an already-tried wrong object), no-lift rate, and controls.

Usage: python icl_clutter.py --expert-checkpoint logs/runs/clutter-expert-s1/final_ckpt.pt --demos 200 --seed 1
"""
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro

import mani_skill.envs  # noqa: F401
from mani_skill.envs.tasks.tabletop.pick_clutter_ycb_cond import K, OBS_DIM, TARGET_SLICE, obs_layout
from ppo_fast import Agent
from run_logging import LOG_ROOT, RunDir

LAY = obs_layout()
S_OBS = TARGET_SLICE.start  # student obs = everything before the target block
TOK = 2 * K + K + 1  # context token dim: final xy per object, lifted flag per object, feedback bit
L_CTX = K - 1  # context = last L_CTX attempts


@dataclass
class Args:
    expert_checkpoint: str = str(LOG_ROOT / "runs/clutter-expert-s1/final_ckpt.pt")
    demos: int = 200  # layouts used for training (each gives K rollouts)
    val_layouts: int = 64
    eval_layouts: int = 512
    num_envs: int = 256
    attempts: int = 6
    steps: int = 20000
    batch: int = 1024
    lr: float = 3e-4
    d: int = 256
    layers: int = 3
    heads: int = 4
    mix: int = 5  # GMM components
    sample: str = "sticky"  # GMM action at rollout: sticky | step | mean
    temp: float = 0.5  # temperature on mixture logits when sampling a component
    icl_samples: str = ""  # extra ICL evals (real ctx) with other samplers, e.g. "step:1.0,mean:1,sticky:1.0"
    dropout: float = 0.0
    bc_onehot_only: bool = False  # BC sees obs[:61] + target one-hot only (no target pose / tcp->target)
    drop_qvel: bool = False  # zero joint velocities in the ICL student's input (copycat test)
    head: str = "gmm"  # gmm | diff (DDPM over the action chunk; needs --arch obj)
    exec_steps: int = 0  # steps executed per predicted chunk before re-planning (0 = whole chunk)
    ddim: int = 10
    chunk: int = 1  # GMM over H-step action chunks, executed open-loop (one mixture sample per chunk)
    arch: str = "flat"  # flat: GMMPolicy over the flat obs; obj: object-centric ObjPolicy
    ctx_path: str = "attn"  # attn: context tokens attended to; cat: flattened context concatenated to the obs input
    ckpt: str = "best"  # weights evaluated after training: best (lowest val NLL) | final
    load_dir: Optional[str] = None  # skip training; load <load_dir>/checkpoints/{bc,icl}[-best].pt
    methods: str = "bc,icl"
    seed: int = 1
    eval_seed: int = 12345
    data_seed: int = 1000
    cache: Optional[str] = None  # task bank cache (.pt); default logs/clutter_bank/<expert>-<demos>-<seed>.pt
    out_dir: Optional[str] = None


# ------------------------------------------------------------------ env / expert
def make_env(n):
    return gym.make("PickClutterYCBCond-v1", num_envs=n, obs_mode="state", control_mode="pd_joint_delta_pos",
                    sim_backend="physx_cuda", robot_init_qpos_noise=0.0)


def load_expert(path, A, dev):
    class _S:
        single_observation_space = type("S", (), {"shape": (OBS_DIM,)})

    ag = Agent(OBS_DIM, A, device=dev) if Agent.__init__.__code__.co_argcount > 2 else Agent(_S)
    ag.load_state_dict(torch.load(path, map_location=dev))
    return ag.eval()


@torch.no_grad()
def run_attempt(env, layout, target, act_fn, T):
    """One attempt on n envs. act_fn(obs (n,OBS_DIM), t) -> action. Returns obs (n,T,OBS_DIM), act (n,T,A), alive (n,T) (steps up
    to and incl. the terminating one), success (n,), lifted_id (n,), token (n,TOK) without feedback."""
    obs, _ = env.reset(options=dict(layout=layout, target=target))
    n = obs.shape[0]
    dev = obs.device
    O, Acts, Al = [], [], []
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    succ = torch.zeros(n, dtype=torch.bool, device=dev)
    lifted = torch.full((n,), -1, dtype=torch.long, device=dev)
    maxlift = torch.zeros(n, K, device=dev)
    fin_pos = torch.zeros(n, K, 2, device=dev)
    for t in range(T):
        a = act_fn(obs, t).clamp(-1, 1)
        O.append(obs), Acts.append(a), Al.append(alive.clone())
        obs, _, _, _, info = env.step(a)
        pos = obs[:, LAY["obj_pose"]].reshape(n, K, 7)[..., :2]
        maxlift = torch.where(alive[:, None], torch.maximum(maxlift, info["obj_lift"]), maxlift)
        fin_pos = torch.where(alive[:, None, None], pos, fin_pos)
        end = alive & (info["success"] | info["fail"])
        succ |= end & info["success"]
        lifted = torch.where(end, info["lifted_id"], lifted)
        alive &= ~end
    tok = torch.cat([fin_pos.reshape(n, -1), (maxlift >= 0.04).float(), torch.zeros(n, 1, device=dev)], 1)
    return dict(obs=torch.stack(O, 1), act=torch.stack(Acts, 1), alive=torch.stack(Al, 1), succ=succ, lifted=lifted, tok=tok)


def collect_bank(env, expert, n_layouts, seed, T, dev):
    """Expert rollouts for every condition on layouts until n_layouts all-K-success layouts. Stored fp16."""
    g = torch.Generator(device=dev).manual_seed(seed)
    N = env.unwrapped.num_envs
    keep = dict(obs=[], act=[], alive=[], tok=[], layout=[])
    have, tried, n_ok, b = 0, 0, 0, 0
    pol = lambda o, t: expert.actor_mean(o)
    while have < n_layouts:
        lay = env.unwrapped.sample_layout(N, gen=g)
        rs = [run_attempt(env, lay, torch.full((N,), c, device=dev), pol, T) for c in range(K)]
        ok = torch.stack([r["succ"] & (r["lifted"] == c) for c, r in enumerate(rs)], 1).all(1)
        idx = ok.nonzero()[:, 0][: n_layouts - have]
        keep["obs"].append(torch.stack([r["obs"][idx] for r in rs], 1).half())  # (n,K,T,OBS)
        keep["act"].append(torch.stack([r["act"][idx] for r in rs], 1))
        keep["alive"].append(torch.stack([r["alive"][idx] for r in rs], 1))
        keep["tok"].append(torch.stack([r["tok"][idx] for r in rs], 1))
        keep["layout"].append(lay[idx])
        have += len(idx)
        tried += N
        n_ok += int(ok.sum())
        b += 1
        print(f"  bank batch {b}: kept {len(idx)}/{N} layouts (all {K} conditions succeed); total {have}/{n_layouts}; "
              f"per-cond succ {[round(r['succ'].float().mean().item(), 3) for r in rs]}", flush=True)
    out = {k: torch.cat(v) for k, v in keep.items()}
    out["keep_frac"] = n_ok / tried  # fraction of sampled layouts where all K conditioned rollouts succeed
    return out


# ------------------------------------------------------------------ model
class GMMPolicy(nn.Module):
    """obs token + up to L_CTX context tokens -> transformer -> GMM over actions (from the obs token)."""

    def __init__(self, obs_dim, A, ctx, d, layers, heads, mix, dropout=0.0, path="attn", drop_qvel=False):
        super().__init__()
        self.drop_qvel = drop_qvel
        self.obs_dim, self.A, self.ctx, self.mix = obs_dim, A, ctx, mix
        self.cat = bool(ctx) and path == "cat"
        if self.cat:
            self.ctx = False  # no context tokens; the context enters through obs_proj
        self.register_buffer("s_mean", torch.zeros(obs_dim)), self.register_buffer("s_std", torch.ones(obs_dim))
        self.register_buffer("c_mean", torch.zeros(TOK)), self.register_buffer("c_std", torch.ones(TOK))
        self.obs_proj = nn.Sequential(nn.Linear(obs_dim + (L_CTX * (TOK + 1) if self.cat else 0), d), nn.GELU(), nn.Linear(d, d))
        if ctx:
            self.tok_proj = nn.Linear(TOK, d)
            self.pos_emb = nn.Parameter(torch.randn(L_CTX, d) * 0.02)  # recency slot (0 = most recent)
        self.obs_emb = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, mix * (2 * A + 1)))

    def set_norm(self, s, c=None, m=None):
        self.s_mean.copy_(s.mean(0)), self.s_std.copy_(s.std(0).clamp_min(1e-3))
        if c is not None and (self.ctx or self.cat):
            c = c[m]
            self.c_mean.copy_(c.mean(0)), self.c_std.copy_(c.std(0).clamp_min(1e-3))
            self.c_mean[2 * K:], self.c_std[2 * K:] = 0.0, 1.0  # binary flags as is

    def forward(self, s, c=None, cmask=None):
        """s (B,obs_dim); c (B,L_CTX,TOK) most recent first; cmask (B,L_CTX) True = present. -> logits (B,M), mu, log_std (B,M,A)."""
        x = (s - self.s_mean) / self.s_std
        if self.drop_qvel:
            x = x.clone()
            x[:, LAY["qvel"]] = 0
        if self.cat:
            cn = ((c - self.c_mean) / self.c_std).clamp(-5, 5) * cmask[..., None]
            x = torch.cat([x, cn.flatten(1), cmask.float()], 1)
        o = self.obs_proj(x) + self.obs_emb
        seq, pad = o[:, None], torch.zeros(len(s), 1, dtype=torch.bool, device=s.device)
        if self.ctx:
            ct = self.tok_proj(((c - self.c_mean) / self.c_std).clamp(-5, 5)) + self.pos_emb
            seq, pad = torch.cat([seq, ct], 1), torch.cat([pad, ~cmask], 1)
        h = self.enc(seq, src_key_padding_mask=pad)[:, 0]
        out = self.head(h)
        M, A = self.mix, self.A
        logits = out[:, :M]
        mu = out[:, M:M + M * A].reshape(-1, M, A)
        log_std = out[:, M + M * A:].reshape(-1, M, A).clamp(-5, 1)
        return logits, mu, log_std

    def nll(self, s, a, c=None, cmask=None):
        logits, mu, ls = self(s, c, cmask)
        lp = (-0.5 * ((a[:, None] - mu) / ls.exp()) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(-1)
        return -(torch.logsumexp(F.log_softmax(logits, -1) + lp, -1)).mean()


def gmm_nll(logits, mu, ls, a):
    lp = (-0.5 * ((a[:, None] - mu) / ls.exp()) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(-1)
    return -(torch.logsumexp(F.log_softmax(logits, -1) + lp, -1)).mean()


class ObjPolicy(nn.Module):
    """Object-centric GMM policy. Tokens: one proprio token (qpos, qvel, tcp pose) + one token per object
    [pose 7, lift, grasped, tcp->obj 3, cond feats] + a learned object-id embedding. cond feats: BC (`mode="bc"`, obs incl.
    target one-hot) -> is-target flag; ICL (`mode="icl"`) -> per context slot [lifted this object, feedback, valid, final xy 2].
    The action GMM is read out from the proprio token."""

    def __init__(self, mode, A, d, layers, heads, mix, dropout=0.0):
        super().__init__()
        self.mode, self.A, self.mix = mode, A, mix
        self.obs_dim = LAY["target_onehot"].stop if mode == "bc" else S_OBS
        self.ctx = mode == "icl"
        self.fo = 12 + (1 if mode == "bc" else 5 * L_CTX)
        self.register_buffer("p_mean", torch.zeros(25)), self.register_buffer("p_std", torch.ones(25))
        self.register_buffer("o_mean", torch.zeros(self.fo)), self.register_buffer("o_std", torch.ones(self.fo))
        self.p_proj = nn.Sequential(nn.Linear(25, d), nn.GELU(), nn.Linear(d, d))
        self.o_proj = nn.Sequential(nn.Linear(self.fo, d), nn.GELU(), nn.Linear(d, d))
        self.id_emb = nn.Parameter(torch.randn(K, d) * 0.02)
        self.p_emb = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, mix * (2 * A + 1)))

    def feats(self, s, c=None, cmask=None):
        B = len(s)
        prop = s[:, :25]
        op = s[:, LAY["obj_pose"]].reshape(B, K, 7)
        f = [op, s[:, LAY["obj_lift"]][..., None], s[:, LAY["obj_grasped"]][..., None], op[..., :3] - s[:, None, 18:21]]
        if self.mode == "bc":
            f.append(s[:, LAY["target_onehot"]][..., None])
        else:
            lifted = c[..., 2 * K:3 * K].transpose(1, 2)  # (B,K,L)
            fb = c[..., -1][:, None].expand(B, K, L_CTX)
            v = cmask.float()[:, None].expand(B, K, L_CTX)
            xy = c[..., :2 * K].reshape(B, L_CTX, K, 2).transpose(1, 2)  # (B,K,L,2)
            f.append(torch.cat([lifted[..., None], fb[..., None], v[..., None], xy], -1).reshape(B, K, 5 * L_CTX) * 1.0)
        return prop, torch.cat(f, -1)

    def set_norm(self, s, c=None, m=None):
        prop, of = self.feats(s, c, m)
        self.p_mean.copy_(prop.mean(0)), self.p_std.copy_(prop.std(0).clamp_min(1e-3))
        of = of.reshape(-1, self.fo)
        self.o_mean.copy_(of.mean(0)), self.o_std.copy_(of.std(0).clamp_min(1e-3))
        flags = [12] if self.mode == "bc" else [12 + 5 * j + i for j in range(L_CTX) for i in range(3)]
        self.o_mean[flags], self.o_std[flags] = 0.0, 1.0  # binary flags as is

    def encode(self, s, c=None, cmask=None):
        prop, of = self.feats(s, c, cmask)
        p = self.p_proj((prop - self.p_mean) / self.p_std) + self.p_emb
        o = self.o_proj(((of - self.o_mean) / self.o_std).clamp(-10, 10)) + self.id_emb
        return self.enc(torch.cat([p[:, None], o], 1))[:, 0]

    def forward(self, s, c=None, cmask=None):
        out = self.head(self.encode(s, c, cmask))
        M, A = self.mix, self.A
        return out[:, :M], out[:, M:M + M * A].reshape(-1, M, A), out[:, M + M * A:].reshape(-1, M, A).clamp(-5, 1)

    def nll(self, s, a, c=None, cmask=None):
        return gmm_nll(*self(s, c, cmask), a)


class DiffObjPolicy(ObjPolicy):
    """ObjPolicy encoder + DDPM head over a flattened action chunk (A here = chunk * action dim). Train: epsilon-prediction
    MSE (`nll` returns it so the training loop is shared). Sample: DDIM with `ddim` steps from Gaussian noise, x0 clipped to
    [-1, 1]; the initial noise is what picks the mode."""

    diff = True

    def __init__(self, mode, A, d, layers, heads, dropout=0.0, T=100, ddim=10, hidden=512):
        super().__init__(mode, A, d, layers, heads, 1, dropout)
        del self.head
        self.T, self.ddim = T, ddim
        t = torch.arange(T + 1) / T
        ab = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        ab = (ab / ab[0])[1:].clamp(1e-5, 0.9999)  # alpha_bar for t = 1..T (index t-1)
        self.register_buffer("ab", ab)
        self.temb = nn.Sequential(nn.Linear(64, d), nn.GELU(), nn.Linear(d, d))
        self.inp = nn.Linear(A + 2 * d, hidden)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)) for _ in range(3))
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, A))

    def _temb(self, t):
        f = torch.exp(torch.arange(32, device=t.device) * (-math.log(1000.0) / 32))
        x = t.float()[:, None] * f
        return self.temb(torch.cat([x.sin(), x.cos()], 1))

    def eps(self, x, t, h):
        z = self.inp(torch.cat([x, self._temb(t), h], 1))
        for b in self.blocks:
            z = z + b(z)
        return self.out(z)

    def nll(self, s, a, c=None, cmask=None):
        h = self.encode(s, c, cmask)
        t = torch.randint(1, self.T + 1, (len(a),), device=a.device)
        ab = self.ab[t - 1][:, None]
        n = torch.randn_like(a)
        return F.mse_loss(self.eps(ab.sqrt() * a + (1 - ab).sqrt() * n, t, h), n)

    @torch.no_grad()
    def sample(self, s, c=None, cmask=None):
        h = self.encode(s, c, cmask)
        x = torch.randn(len(s), self.A, device=s.device)
        ts = torch.linspace(self.T, 1, self.ddim, device=s.device).round().long()
        for i, t in enumerate(ts):
            tt = t.expand(len(s))
            ab = self.ab[t - 1]
            e = self.eps(x, tt, h)
            x0 = ((x - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-1, 1)
            if i + 1 < len(ts):
                abn = self.ab[ts[i + 1] - 1]
                e = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
                x = abn.sqrt() * x0 + (1 - abn).sqrt() * e
            else:
                x = x0
        return x


class Sampler:
    """Picks the GMM action. sticky: keep the previous component while its weight >= 0.2, else resample (temp); step: resample
    each step; mean: mixture mean."""

    def __init__(self, mode, temp, n, dev):
        self.mode, self.temp, self.k = mode, temp, torch.full((n,), -1, dtype=torch.long, device=dev)

    def reset(self):
        self.k.fill_(-1)

    def __call__(self, logits, mu):
        w = logits.softmax(-1)
        if self.mode == "mean":
            return (w[..., None] * mu).sum(1)
        new = torch.distributions.Categorical(logits=logits / self.temp).sample()
        if self.mode == "sticky":
            keep = (self.k >= 0) & (w.gather(1, self.k.clamp_min(0)[:, None])[:, 0] >= 0.2)
            new = torch.where(keep, self.k, new)
        self.k = new
        return mu[torch.arange(len(mu), device=mu.device), new]


# ------------------------------------------------------------------ ICL samples
def search_contexts(target, gen):
    """Random searcher history for each target: random permutation tried in order until the target, then the target again.
    Returns ids (B,L_CTX) of the last <=L_CTX attempts (most recent first; -1 = absent) and feedback (B,L_CTX)."""
    B, dev = len(target), target.device
    perm = torch.argsort(torch.rand(B, K, generator=gen, device=dev), 1)
    pos_t = (perm == target[:, None]).float().argmax(1)  # attempts needed to hit t (0-based)
    h = torch.randint(0, K + 2, (B,), generator=gen, device=dev)  # history length 0..K+1
    ids = torch.full((B, L_CTX), -1, dtype=torch.long, device=dev)
    for j in range(L_CTX):
        a = h - 1 - j  # attempt index of slot j
        ok = a >= 0
        obj = torch.where(a <= pos_t, perm.gather(1, a.clamp(0, K - 1)[:, None])[:, 0], target)
        ids[:, j] = torch.where(ok, obj, -1)
    fb = (ids == target[:, None]) & (ids >= 0)
    return ids, fb


def build_ctx(tok, lid, ids, fb):
    """tok (Nl,K,TOK) bank tokens; lid (B,) layout ids; ids/fb (B,L_CTX). -> c (B,L_CTX,TOK), mask."""
    c = tok[lid[:, None], ids.clamp_min(0)].clone()
    c[..., -1] = fb.float()
    mask = ids >= 0
    c[~mask] = 0
    return c, mask


# ------------------------------------------------------------------ training
def train(model, bank, val, args, dev, run, tag, ctx):
    obs, act, alive, tok = bank["obs"], bank["act"], bank["alive"], bank["tok"]
    Nl, _, T, _ = obs.shape
    idx = alive.nonzero()  # (n,3): layout, cond, t
    v_idx = val["alive"].nonzero()
    sdim = model.obs_dim
    g = torch.Generator(device=dev).manual_seed(args.seed)
    s_all = obs[idx[:, 0], idx[:, 1], idx[:, 2]].float()[:, :sdim]
    if ctx:
        ids, fb = search_contexts(idx[:, 1], g)
        c, m = build_ctx(tok, idx[:, 0], ids, fb)
        model.set_norm(s_all, c, m)
    else:
        model.set_norm(s_all)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    def batch(src, ix, gen):
        sel = ix[torch.randint(0, len(ix), (args.batch,), generator=gen, device=dev)]
        lid, cond, t = sel[:, 0], sel[:, 1], sel[:, 2]
        s = src["obs"][lid, cond, t].float()[:, :sdim]
        H = args.chunk
        last = src["alive"][lid, cond].sum(-1) - 1  # last alive step
        tt = torch.minimum(t[:, None] + torch.arange(H, device=dev), last[:, None])  # pad past the end with the last action
        a = src["act"][lid[:, None], cond[:, None], tt].reshape(len(t), -1)
        if not ctx:
            return s, a, None, None
        ids, fb = search_contexts(cond, gen)
        c, m = build_ctx(src["tok"], lid, ids, fb)
        return s, a, c, m

    gv = torch.Generator(device=dev).manual_seed(0)
    vb = [batch(val, v_idx, gv) for _ in range(8)]
    best, t0 = 1e9, time.time()
    for step in range(1, args.steps + 1):
        loss = model.nll(*batch(bank, idx, g))
        opt.zero_grad(), loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step()
        if step % 500 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vl = sum(model.nll(*b).item() for b in vb) / len(vb)
            model.train()
            run.log(step, {f"{tag}/train_nll": loss.item(), f"{tag}/val_nll": vl})
            if vl < best:
                best = vl
                torch.save(model.state_dict(), run.ckpt(f"{tag}-best.pt"))
            print(f"[{tag}] step {step} train {loss.item():.3f} val {vl:.3f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), run.ckpt(f"{tag}.pt"))
    return best


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(env, layouts, targets, policy_fn, attempts, T, dev, ctx_mode="real"):
    """Run `attempts` attempts per eval layout. policy_fn(obs, t, c, cmask, k) -> action. ctx_mode: real | none | shuf_fb.
    Returns per-attempt arrays."""
    N = layouts.shape[0]
    hist_tok = torch.zeros(N, 0, TOK, device=dev)
    hist_ids = torch.zeros(N, 0, dtype=torch.long, device=dev)
    hist_fb = torch.zeros(N, 0, dtype=torch.bool, device=dev)
    res = dict(succ=[], lifted=[], repeat=[], nolift=[])
    found = torch.zeros(N, dtype=torch.bool, device=dev)
    for k in range(attempts):
        h = hist_tok.shape[1]
        c = torch.zeros(N, L_CTX, TOK, device=dev)
        m = torch.zeros(N, L_CTX, dtype=torch.bool, device=dev)
        if ctx_mode != "none":
            for j in range(min(h, L_CTX)):
                c[:, j] = hist_tok[:, h - 1 - j]
                m[:, j] = True
            if ctx_mode == "shuf_fb":
                c[..., -1] = (torch.rand(N, L_CTX, device=dev) < 0.5).float() * m
        r = run_attempt(env, layouts, targets, lambda o, t: policy_fn(o, t, c, m, k), T)
        tried_wrong = ((hist_ids == r["lifted"][:, None]) & ~hist_fb).any(1) if h else torch.zeros(N, dtype=torch.bool, device=dev)
        res["succ"].append(r["succ"].float().mean().item())
        res["nolift"].append((r["lifted"] < 0).float().mean().item())
        res["repeat"].append((tried_wrong & (r["lifted"] >= 0)).float().mean().item())
        res["lifted"].append(r["lifted"].cpu())
        found |= r["succ"]
        res.setdefault("pass", []).append(found.float().mean().item())
        tk = r["tok"].clone()
        tk[:, -1] = r["succ"].float()
        hist_tok = torch.cat([hist_tok, tk[:, None]], 1)
        hist_ids = torch.cat([hist_ids, r["lifted"][:, None]], 1)
        hist_fb = torch.cat([hist_fb, r["succ"][:, None]], 1)
    res["lifted"] = torch.stack(res["lifted"], 1).tolist()
    return res


def main():
    args = tyro.cli(Args)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    out = args.out_dir or str(LOG_ROOT / f"runs/icl-clutter-d{args.demos}-s{args.seed}")
    run = RunDir(out, args)
    env = make_env(args.num_envs)
    T = env.unwrapped.spec.max_episode_steps or 75
    A = env.action_space.shape[-1]
    expert = load_expert(args.expert_checkpoint, A, dev)

    ename = os.path.basename(os.path.dirname(args.expert_checkpoint)) + "-" + os.path.basename(args.expert_checkpoint)[:-3]
    cache = args.cache or str(LOG_ROOT / f"clutter_bank/{ename}-d{args.demos}-v{args.val_layouts}-ds{args.data_seed}.pt")
    if os.path.exists(cache):
        bank, val = torch.load(cache, map_location=dev)
        print(f"loaded bank {cache}")
    else:
        print("collecting expert bank ...")
        bank = collect_bank(env, expert, args.demos, args.data_seed, T, dev)
        val = collect_bank(env, expert, args.val_layouts, args.data_seed + 7777, T, dev)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save((bank, val), cache)
    print(f"bank: {bank['obs'].shape[0]} layouts, keep frac {bank['keep_frac']:.3f}")

    # eval tasks: held-out layouts + targets, processed in chunks of num_envs
    ge = torch.Generator(device=dev).manual_seed(args.eval_seed)
    E, N = args.eval_layouts, args.num_envs
    ev_lay = torch.cat([env.unwrapped.sample_layout(N, gen=ge) for _ in range(math.ceil(E / N))])[:E]
    ev_tgt = torch.randint(0, K, (E,), generator=ge, device=dev)

    def eval_all(policy_fn, attempts, ctx_mode="real", sampler=None):
        parts = []
        for i in range(0, E, N):
            if sampler is not None:
                sampler.reset()
            parts.append(evaluate(env, ev_lay[i:i + N], ev_tgt[i:i + N], policy_fn, attempts, T, dev, ctx_mode))
        w = [min(N, E - i) / E for i in range(0, E, N)]
        return {k: ([sum(wi * p[k][j] for wi, p in zip(w, parts)) for j in range(attempts)] if k != "lifted" else sum((p[k] for p in parts), []))
                for k in parts[0]}

    results = dict(keep_frac=bank["keep_frac"], n_layouts=bank["obs"].shape[0])
    expert_fn = lambda o, t, c, m, k: expert.actor_mean(o)
    results["expert"] = eval_all(expert_fn, 1)
    print("expert attempt-1 succ", results["expert"]["succ"])

    methods = args.methods.split(",")
    samp = Sampler(args.sample, args.temp, N, dev)

    def gmm_fn(model, sdim, ctx, rand_cond=False):
        def f(o, t, c, m, k):
            if t == 0:
                samp.reset()
                if rand_cond:
                    f.cond = torch.randint(0, K, (o.shape[0],), device=dev)
            s = o[:, :sdim]
            if rand_cond:
                s = s.clone()
                oh = F.one_hot(f.cond, K).float()
                s[:, LAY["target_onehot"]] = oh
                tp = o[:, LAY["obj_pose"]].reshape(-1, K, 7)[torch.arange(len(o), device=dev), f.cond]
                if sdim == OBS_DIM:
                    s[:, LAY["target_pose"]] = tp
                    s[:, LAY["tcp_to_target"]] = tp[:, :3] - o[:, LAY["tcp_pose"]][:, :3]
            H = args.chunk
            E = args.exec_steps or H
            if t % E == 0:
                if getattr(model, "diff", False):
                    f.buf = model.sample(s, c if ctx else None, m if ctx else None).reshape(len(s), H, A)
                else:
                    logits, mu, _ = model(s, c if ctx else None, m if ctx else None)
                    f.buf = samp(logits, mu).reshape(len(s), H, A)
            return f.buf[:, t % E]
        return f

    def fit(model, tag, ctx):
        if args.load_dir is None:
            results[f"{tag}_val_nll"] = train(model, bank, val, args, dev, run, tag, ctx=ctx)
            src = run.ckpt(f"{tag}-best.pt" if args.ckpt == "best" else f"{tag}.pt")
        else:
            src = f"{args.load_dir}/checkpoints/{tag}-best.pt" if args.ckpt == "best" else f"{args.load_dir}/checkpoints/{tag}.pt"
        model.load_state_dict(torch.load(src, map_location=dev))
        print(f"[{tag}] evaluating {src}")
        return model.eval()

    def show(name):
        r = results[name]
        print(f"{name}: pass {[round(x, 3) for x in r['pass']]} succ {[round(x, 3) for x in r['succ']]} "
              f"repeat {[round(x, 3) for x in r['repeat']]} nolift {[round(x, 3) for x in r['nolift']]}", flush=True)

    if "bc" in methods:
        BD = LAY["target_onehot"].stop if args.bc_onehot_only else OBS_DIM
        if args.arch == "obj":
            BD = LAY["target_onehot"].stop
            bc = fit((DiffObjPolicy("bc", A * args.chunk, args.d, args.layers, args.heads, args.dropout, ddim=args.ddim) if args.head == "diff"
                      else ObjPolicy("bc", A * args.chunk, args.d, args.layers, args.heads, args.mix, args.dropout)).to(dev), "bc", False)
        else:
            bc = fit(GMMPolicy(BD, A * args.chunk, False, args.d, args.layers, args.heads, args.mix, args.dropout).to(dev), "bc", False)
        results["bc"] = eval_all(gmm_fn(bc, BD, False), args.attempts, sampler=samp)
        results["bc_guess"] = eval_all(gmm_fn(bc, BD, False, rand_cond=True), args.attempts, sampler=samp)
        show("bc"), show("bc_guess")
    if "icl" in methods:
        if args.arch == "obj":
            icl = fit((DiffObjPolicy("icl", A * args.chunk, args.d, args.layers, args.heads, args.dropout, ddim=args.ddim) if args.head == "diff"
                       else ObjPolicy("icl", A * args.chunk, args.d, args.layers, args.heads, args.mix, args.dropout)).to(dev), "icl", True)
        else:
            icl = fit(GMMPolicy(S_OBS, A * args.chunk, True, args.d, args.layers, args.heads, args.mix, args.dropout, args.ctx_path, args.drop_qvel).to(dev), "icl", True)
        for mode in ("real", "none", "shuf_fb"):
            results[f"icl_{mode}"] = eval_all(gmm_fn(icl, S_OBS, True), args.attempts, ctx_mode=mode, sampler=samp)
            show(f"icl_{mode}")
        for spec in filter(None, args.icl_samples.split(",")):
            mode, temp = spec.split(":")
            samp.mode, samp.temp = mode, float(temp)
            results[f"icl_real_{mode}{temp}"] = eval_all(gmm_fn(icl, S_OBS, True), args.attempts, sampler=samp)
            show(f"icl_real_{mode}{temp}")
        samp.mode, samp.temp = args.sample, args.temp
    es = results["expert"]["succ"][0]
    results["oracle_pass"] = [es * min(1.0, (k + 1) / K) for k in range(args.attempts)]
    results["guess_pass_ref"] = [es * (1 - (1 - 1 / K) ** (k + 1)) for k in range(args.attempts)]
    run.save_json("results.json", results)
    print("done", out)


if __name__ == "__main__":
    main()
