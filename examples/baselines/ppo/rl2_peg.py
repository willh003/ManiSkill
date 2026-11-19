"""Recurrent PPO (RL^2-style) on MetaPeg: hidden hole, n_ep episodes per task, feedback c = hole - f after each episode.
Plan: docs/claude/ideas/2026-11-17-rl2-peg-upper-bound-plan.md.

Phase 1 (BC init, DAgger): roll out a beta-mix of the student and the scripted oracle (ep 1 = prior-mean-hole expert, ep k>1 = expert on
  the mean of f_j + c_j); labels = expert on the true hole (ep >= 2) and, by default, the prior-mean-hole expert in ep 1 (--ep1-label prior;
  'true' gives the expert-on-true-hole label, i.e. the posterior-mean action). Aggregated data, full-sequence BPTT.
Phase 2 (PPO): GRU actor (actor obs only), feedforward privileged critic (actor obs + true clean obs). One rollout = one meta-episode
  (n_ep * 100 steps, lockstep); GAE runs across episode boundaries (done only at the meta-episode end). Value warm-up with the actor frozen,
  optional auxiliary BC loss on the oracle labels annealed to 0.
Eval: deterministic, on the 512 fixed eval tasks (wide_corner 'all', eval_seed 12345; tiled if num_envs > 512, primary metric = first 512),
  controls at the end: shuffled (f, c) across tasks, zeroed feedback, hidden state reset at each episode start.
Resumable: checkpoints/latest.pt (PPO phase) and bc.pt."""
import json
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import tyro

from icl_tasks import checker_sample
from meta_peg import EP_LEN, S_DIM, MetaPeg, believed_hole_obs, load_expert, prior_hole_obs
from run_logging import LOG_ROOT, RunDir


@dataclass
class Args:
    expert_checkpoint: str = str(LOG_ROOT / "runs/peg-expert-r1/ckpt_3576.pt")
    out_dir: str = str(LOG_ROOT / "runs/rl2-peg")
    seed: int = 1
    num_envs: int = 1024
    n_ep: int = 2
    fb_noise_xy: float = 0.0
    fb_noise_yaw: float = 0.0
    split: str = "wide_corner"
    train_region: str = "all"  # "all" | "white": train (and eval ID) on this region of --split; "white" also evals OOD on the complement ("black")
    same_init: bool = True
    hide_f: bool = False  # do not give the final peg pose f in the feedback (only c); the policy must remember it from the last frames
    full_obs: bool = False  # oracle baseline: actor sees the true hole (use --n-ep 1 --ep1-label true)
    memoryless: bool = False  # actor ignores the GRU (feedforward on the same obs); baseline for the memory ablation
    eval_seed: int = 12345
    hidden: int = 256
    # BC
    bc_iters: int = 12
    bc_beta_iters: int = 4  # beta (oracle share) decays 1 -> 0 over these iterations
    bc_steps: int = 1000  # grad steps per BC iteration
    bc_batch: int = 128  # sequences per grad step
    bc_lr: float = 1e-3
    ep1_label: str = "prior"  # prior | true
    init: str = ""  # start from this bc.pt / latest.pt instead of running BC
    # PPO
    ppo_iters: int = 300
    value_warmup: int = 10
    lr_actor: float = 1e-4
    lr_critic: float = 3e-4
    init_logstd: float = -1.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 4
    num_minibatches: int = 8
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.05
    bc_coef: float = 0.0  # auxiliary BC loss weight at iteration 0, annealed linearly to 0 over bc_anneal iterations
    bc_anneal: int = 100
    w_ep1: float = 1.0  # reward weight of episode 1 (later episodes: 1)
    eval_every: int = 10


def mlp(i, o, h=256, n=2, out_std=1.0):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    last = nn.Linear(d, o)
    nn.init.orthogonal_(last.weight, out_std), nn.init.zeros_(last.bias)
    return nn.Sequential(*layers, last)


class Norm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.register_buffer("mean", torch.zeros(d)), self.register_buffer("std", torch.ones(d))

    def set(self, x):
        x = x.reshape(-1, x.shape[-1])
        self.mean.copy_(x.mean(0)), self.std.copy_(x.std(0).clamp_min(1e-2))

    def forward(self, x):
        return ((x - self.mean) / self.std).clamp(-10, 10)


class Actor(nn.Module):
    """enc -> GRU; head on [enc, gru] (skip path). forward(x (B,T,D), h (1,B,H)) -> mean (B,T,A), h."""

    def __init__(self, D, A, H, memoryless=False):
        super().__init__()
        self.memoryless = memoryless
        self.norm = Norm(D)
        self.enc = nn.Sequential(nn.Linear(D, H), nn.Tanh(), nn.Linear(H, H), nn.Tanh())
        self.gru = nn.GRU(H, H, batch_first=True)
        self.head = mlp(2 * H, A, H, 1, out_std=0.01)
        self.logstd = nn.Parameter(torch.zeros(A))
        self.H = H

    def forward(self, x, h=None):
        e = self.enc(self.norm(x))
        if self.memoryless:
            return self.head(torch.cat([e, torch.zeros_like(e)], -1)), h
        g, h = self.gru(e, h)
        return self.head(torch.cat([e, g], -1)), h


class Critic(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.norm = Norm(D)
        self.net = mlp(D, 1, H, 3)

    def forward(self, x):
        return self.net(self.norm(x)).squeeze(-1)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.dev = torch.device("cuda")
        torch.manual_seed(args.seed)
        self.env = MetaPeg(args.num_envs, args.n_ep, args.fb_noise_xy, args.fb_noise_yaw, args.split, args.same_init, hide_f=args.hide_f, full_obs=args.full_obs, train_region=args.train_region)
        self.expert = load_expert(args.expert_checkpoint, self.dev)
        self.N, self.T, self.A = args.num_envs, self.env.T, self.env.A
        self.actor = Actor(self.env.actor_dim, self.A, args.hidden, args.memoryless).to(self.dev)
        self.critic = Critic(self.env.critic_dim, args.hidden).to(self.dev)
        g = torch.Generator(device=self.dev).manual_seed(args.eval_seed)
        eg = checker_sample(512, args.train_region, self.dev, g, args.split)
        self.eval_goals = eg.repeat(math.ceil(self.N / 512), 1)[: self.N]
        self.ood_goals = None
        if args.train_region != "all":
            og = checker_sample(512, "black", self.dev, g, args.split)
            self.ood_goals = og.repeat(math.ceil(self.N / 512), 1)[: self.N]
        self.gen = torch.Generator(device=self.dev).manual_seed(1000 + args.seed)
        self.seed_ctr = 10 ** 6 * args.seed

    def next_seed(self):
        self.seed_ctr += 1
        return self.seed_ctr

    # ---- labels / oracle
    def label(self):
        o = self.env.expert_obs()
        x = prior_hole_obs(o) if (self.env.ep == 0 and self.args.ep1_label == "prior") else o
        return self.expert.get_action(x, deterministic=True)

    def oracle_action(self):
        o = self.env.expert_obs()
        if self.env.ep == 0:
            return self.expert.get_action(prior_hole_obs(o), deterministic=True)
        est = torch.stack([f + c for f, c in zip(self.env.f_hist, self.env.c_hist)]).mean(0)
        return self.expert.get_action(believed_hole_obs(o, est), deterministic=True)

    # ---- rollouts
    @torch.no_grad()
    def run_episode(self, policy="student", beta=0.0, goals=None, seed=None, stochastic=False, ctrl="", record=False):
        """One meta-episode. policy: student | oracle. beta: per-env prob of the oracle (DAgger mix). Returns dict."""
        seed = self.next_seed() if seed is None else seed
        a_obs, c_obs = self.env.reset(seed, goals, self.gen)
        use_or = torch.rand(self.N, device=self.dev, generator=self.gen) < beta
        h = None
        rec = dict(a=[], c=[], act=[], lab=[], logp=[], val=[], rew=[], mean=[])
        perm = torch.randperm(self.N, device=self.dev)
        for t in range(self.T):
            if self.env.t == 0 and ctrl == "hreset":
                h = None
            x = a_obs
            if ctrl in ("shuf", "zero") and self.env.ep > 0:
                x = x.clone()
                fc = slice(S_DIM + self.env.n_ep + 1, S_DIM + self.env.n_ep + 1 + 14)
                x[:, fc] = x[perm, fc] if ctrl == "shuf" else 0
                if ctrl == "zero":
                    x[:, -1] = 0
            if policy == "student" or beta > 0:
                mean, h = self.actor(x[:, None], h)
                mean = mean[:, 0]
                act = mean + self.actor.logstd.exp() * torch.randn_like(mean) if stochastic else mean
            if policy == "oracle" or beta > 0:
                oa = self.oracle_action()
                act = oa if policy == "oracle" else torch.where(use_or[:, None], oa, act)
            if record:
                rec["a"].append(x), rec["c"].append(c_obs), rec["act"].append(act), rec["lab"].append(self.label())
                if stochastic:
                    std = self.actor.logstd.exp()
                    rec["logp"].append(torch.distributions.Normal(mean, std).log_prob(act).sum(-1))
                    rec["val"].append(self.critic(c_obs)), rec["mean"].append(mean)
            ep = self.env.ep
            a_obs, c_obs, rew, done, info = self.env.step(act)
            if record:
                rec["rew"].append(rew * (self.args.w_ep1 if ep == 0 else 1.0))
        out = {k: torch.stack(v, 1) for k, v in rec.items() if v}
        out["succ"] = self.env.succ.clone()
        return out

    def eval(self, ctrl=""):
        r = self.run_episode("student", goals=self.eval_goals, seed=self.args.eval_seed, ctrl=ctrl)
        s = r["succ"].float()
        d = {}
        for name, ss in (("", s[:512]), ("all_", s)):
            for k in range(self.env.n_ep):
                d[f"{name}succ_ep{k + 1}"] = ss[:, k].mean().item()
            d[f"{name}pass@{self.env.n_ep}"] = ss.max(1).values.mean().item()
        if self.ood_goals is not None and not ctrl:
            so = self.run_episode("student", goals=self.ood_goals, seed=self.args.eval_seed)["succ"].float()[:512]
            for k in range(self.env.n_ep):
                d[f"ood_succ_ep{k + 1}"] = so[:, k].mean().item()
            d[f"ood_pass@{self.env.n_ep}"] = so.max(1).values.mean().item()
        return d

    # ---- BC
    def bc(self, rd):
        a = self.args
        opt = torch.optim.Adam(list(self.actor.enc.parameters()) + list(self.actor.gru.parameters()) + list(self.actor.head.parameters()), lr=a.bc_lr)
        X, Y = [], []
        for it in range(a.bc_iters):
            beta = max(0.0, 1 - it / max(a.bc_beta_iters, 1))
            r = self.run_episode("student", beta=beta, record=True)
            if it == 0:
                self.actor.norm.set(r["a"]), self.critic.norm.set(r["c"])
            X.append(r["a"].half()), Y.append(r["lab"].half())
            Xc, Yc = torch.cat(X), torch.cat(Y)
            M, losses = len(Xc), []
            for _ in range(a.bc_steps):
                idx = torch.randint(0, M, (a.bc_batch,), device=self.dev)
                pred, _ = self.actor(Xc[idx].float())
                loss = ((pred - Yc[idx].float()) ** 2).mean()
                opt.zero_grad(), loss.backward(), nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0), opt.step()
                losses.append(loss.item())
            # per-episode loss on the latest (on-policy-ish) batch
            with torch.no_grad():
                pred, _ = self.actor(r["a"])
                e = ((pred - r["lab"]) ** 2).mean(-1)
            ev = self.eval()
            m = dict(phase_bc=1, beta=beta, bc_loss=sum(losses[-50:]) / 50, rollout_succ_ep1=r["succ"][:, 0].float().mean().item(),
                     **{f"label_mse_ep{k + 1}": e[:, k * EP_LEN:(k + 1) * EP_LEN].mean().item() for k in range(self.env.n_ep)},
                     **{f"eval_{k}": v for k, v in ev.items()})
            rd.log(it, m)
            print(f"[bc {it}] " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
        torch.save(dict(actor=self.actor.state_dict(), critic=self.critic.state_dict()), rd.ckpt("bc.pt"))

    # ---- PPO
    def ppo(self, rd, start_it=0, opt_state=None):
        a = self.args
        opt = torch.optim.Adam([dict(params=[p for n, p in self.actor.named_parameters()], lr=a.lr_actor),
                                dict(params=self.critic.parameters(), lr=a.lr_critic)], eps=1e-5)
        if opt_state is not None:
            opt.load_state_dict(opt_state)
        N, T = self.N, self.T
        for it in range(start_it, a.ppo_iters):
            t0 = time.time()
            r = self.run_episode("student", stochastic=True, record=True)
            # GAE (terminal at the meta-episode end)
            adv = torch.zeros(N, T, device=self.dev)
            last = torch.zeros(N, device=self.dev)
            for t in reversed(range(T)):
                nv = r["val"][:, t + 1] if t + 1 < T else torch.zeros(N, device=self.dev)
                delta = r["rew"][:, t] + a.gamma * nv - r["val"][:, t]
                last = delta + a.gamma * a.gae_lambda * last
                adv[:, t] = last
            ret = adv + r["val"]
            warm = it < a.value_warmup
            bc_w = a.bc_coef * max(0.0, 1 - it / max(a.bc_anneal, 1))
            mb = N // a.num_minibatches
            stats = dict(pg=[], vf=[], kl=[], clipfrac=[], bc=[])
            stop = False
            for ep in range(a.epochs):
                perm = torch.randperm(N, device=self.dev)
                for i in range(a.num_minibatches):
                    idx = perm[i * mb:(i + 1) * mb]
                    v = self.critic(r["c"][idx])
                    vf = 0.5 * ((v - ret[idx]) ** 2).mean()
                    loss = a.vf_coef * vf
                    if not warm:
                        mean, _ = self.actor(r["a"][idx])
                        dist = torch.distributions.Normal(mean, self.actor.logstd.exp())
                        lp = dist.log_prob(r["act"][idx]).sum(-1)
                        lr_ = lp - r["logp"][idx]
                        ratio = lr_.exp()
                        ad = adv[idx]
                        ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                        pg = torch.max(-ad * ratio, -ad * ratio.clamp(1 - a.clip, 1 + a.clip)).mean()
                        ent = dist.entropy().sum(-1).mean()
                        bcl = ((mean - r["lab"][idx]) ** 2).mean()
                        loss = loss + pg - a.ent_coef * ent + bc_w * bcl
                        with torch.no_grad():
                            kl = ((ratio - 1) - lr_).mean().item()
                        stats["pg"].append(pg.item()), stats["kl"].append(kl), stats["bc"].append(bcl.item())
                        stats["clipfrac"].append(((ratio - 1).abs() > a.clip).float().mean().item())
                    stats["vf"].append(vf.item())
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), a.max_grad_norm)
                    if warm:
                        for p in self.actor.parameters():
                            p.grad = None
                    opt.step()
                    if not warm and a.target_kl and kl > a.target_kl:
                        stop = True
                        break
                if stop:
                    break
            s = r["succ"].float()
            m = dict(phase_bc=0, iter_time=time.time() - t0, rew_ep1=r["rew"][:, :EP_LEN].sum(1).mean().item(),
                     rew_last=r["rew"][:, -EP_LEN:].sum(1).mean().item(), logstd=self.actor.logstd.mean().item(), bc_w=bc_w,
                     epochs_done=ep + 1, **{f"train_succ_ep{k + 1}": s[:, k].mean().item() for k in range(self.env.n_ep)},
                     **{k: sum(v) / len(v) for k, v in stats.items() if v})
            if (it + 1) % a.eval_every == 0 or it == a.ppo_iters - 1:
                m.update({f"eval_{k}": v for k, v in self.eval().items()})
                torch.save(dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), opt=opt.state_dict(), it=it + 1), rd.ckpt("latest.pt"))
                torch.save(dict(actor=self.actor.state_dict(), critic=self.critic.state_dict()), rd.ckpt(f"it{it + 1}.pt"))
            step = a.bc_iters + it
            rd.log(step, m)
            print(f"[ppo {it}] " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)


def main(args):
    rd = RunDir(args.out_dir, args)
    tr = Trainer(args)
    o = tr.run_episode("oracle", goals=tr.eval_goals, seed=args.eval_seed)["succ"].float()
    ref = dict(oracle_succ=o[:512].mean(0).tolist(), oracle_pass=o[:512].max(1).values.mean().item())
    print("oracle (this env size):", ref, flush=True)
    latest = rd.ckpt("latest.pt")
    start, opt_state = 0, None
    if os.path.exists(latest):
        ck = torch.load(latest, map_location=tr.dev)
        tr.actor.load_state_dict(ck["actor"]), tr.critic.load_state_dict(ck["critic"])
        start, opt_state = ck["it"], ck["opt"]
        print(f"resumed from {latest} at iteration {start}", flush=True)
    else:
        if args.init:
            ck = torch.load(args.init, map_location=tr.dev)
            tr.actor.load_state_dict(ck["actor"]), tr.critic.load_state_dict(ck["critic"])
        elif os.path.exists(rd.ckpt("bc.pt")):
            ck = torch.load(rd.ckpt("bc.pt"), map_location=tr.dev)
            tr.actor.load_state_dict(ck["actor"]), tr.critic.load_state_dict(ck["critic"])
        else:
            tr.bc(rd)
        with torch.no_grad():
            tr.actor.logstd.fill_(args.init_logstd)
        ref["bc_eval"] = tr.eval()
        print("BC init eval:", ref["bc_eval"], flush=True)
    tr.ppo(rd, start, opt_state)
    res = dict(ref=ref, final=tr.eval(), controls={c: tr.eval(c) for c in ("shuf", "zero", "hreset")})
    res["succ_matrix"] = tr.run_episode("student", goals=tr.eval_goals, seed=args.eval_seed)["succ"][:512].int().tolist()  # (512 tasks, n_ep)
    if tr.ood_goals is not None:
        res["ood_succ_matrix"] = tr.run_episode("student", goals=tr.ood_goals, seed=args.eval_seed)["succ"][:512].int().tolist()
    rd.save_json("results.json", res)
    print(json.dumps(res, indent=1))
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
