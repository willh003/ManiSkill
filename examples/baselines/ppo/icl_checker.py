"""Checkerboard hold-out test on PegInsertionSideBias-v1: ICL student with per-step feedback vs BC (docs/claude/implementations/checkerboard-holdout.md).

Hole (box) poses live on a 2x2 checkerboard (icl_tasks.checker_sample). Training data only on white cells; evaluation on white and black.
All policies see the true hole pose (obs dims 35:42) at test time; the student obs has one extra flag dim (1 = hole pose masked).

--mode icl: per iteration, N tasks with a true white goal.
  attempt 1: the expert acts on a random wrong hole pose (any cell); per-step feedback = (hole - peg head, head, cos/sin hole yaw) with the TRUE hole.
  attempt 2: the student (hole pose masked at rollout) conditioned on attempt 1 (every ctx_stride-th step: state, executed action, feedback).
  Both attempts are stored with the TRUE hole pose in the obs and relabelled by the expert; in training each sample (query + context) has its hole
  pose masked with probability --pose-dropout. 2N trajectories / iteration. Queries: attempt 1 with h=0, attempt 2 with h in {0, 1}.
--mode bc: per iteration 2N clean expert demos on white goals (true pose, never masked), no context. Same network and grad steps.
Eval (white and black, 512 fixed goals each): pass@1 = success_once of attempt 1; pass@2 = success_once in attempt 1 or 2.
ICL attempt 2 is conditioned on attempt 1 (h=1); for BC (and the no_ctx control) attempt 2 is an independent reset of the same goal.
"""
import time
from dataclasses import dataclass

import gymnasium as gym
import matplotlib
import torch
import torch.nn as nn
import tyro

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mani_skill.envs  # noqa: F401
from mani_skill.envs.tasks.tabletop.peg_insertion_side_bias import peg_head_pos, quaternion_multiply
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from icl_tasks import PegTask, checker_sample, yaw_of_quat
from ppo import Agent
from run_logging import LOG_ROOT, RunDir

TASK = PegTask(full_info=True)  # student sees the true hole pose; the box pose is fixed per task
POSE = slice(35, 42)  # hole pose inside the student obs
S_DIM, Y_DIM = 43, 8  # student obs (42 + mask flag), per-step feedback


@dataclass
class Args:
    mode: str = "icl"
    """icl | bc"""
    expert_checkpoint: str = str(LOG_ROOT / "runs/peg-expert-r1/ckpt_3576.pt")
    exp_name: str = "checker"
    out_dir: str = ""
    num_envs: int = 512
    num_steps: int = 100
    num_iterations: int = 30
    grad_steps: int = 2000
    batch_size: int = 256
    lr: float = 3e-4
    ctx_stride: int = 5
    pose_dropout: float = 0.5
    """icl: probability that a training sample (query + context) has its hole pose masked"""
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    seed: int = 1
    eval_seed: int = 12345
    split: str = "checker"
    """hold-out split name (icl_tasks.SPLITS); white = train region, black = test region"""
    eval_envs_only: bool = False
    """only evaluate the expert references and exit (smoke test)"""


def sobs(obs, masked=False):
    o = TASK.student_obs(obs).clone()
    if masked:
        o[..., POSE] = 0
    return torch.cat([o, torch.full_like(o[..., :1], float(masked))], -1)


def apply_mask(o, m):
    """o (B, ..., 43) with the flag last, m (B,) bool: zero the hole pose and set the flag where m."""
    m_ = m.view(-1, *[1] * (o.dim() - 1))
    o = o.clone()
    o[..., POSE] = torch.where(m_, torch.zeros_like(o[..., POSE]), o[..., POSE])
    o[..., 42:43] = m_.float().expand_as(o[..., 42:43])
    return o


def step_feedback(obs):
    head = peg_head_pos(obs)
    yaw = yaw_of_quat(obs[..., 38:42])
    return torch.cat([obs[..., 35:38] - head, head, yaw.cos()[..., None], yaw.sin()[..., None]], -1)


def wrong_expert_obs(obs, goal, fake):
    """Expert obs whose hole pose is shifted / rotated from the true box pose `goal` to the (wrong) box pose `fake` (both (N, 3): x, y, yaw)."""
    o = obs[..., :43].clone()
    o[..., 35:37] += fake[:, :2] - goal[:, :2]
    half = (fake[:, 2] - goal[:, 2]) / 2
    z = torch.zeros_like(half)
    o[..., 38:42] = quaternion_multiply(torch.stack([half.cos(), z, z, half.sin()], -1), o[..., 38:42])
    return o


class Policy(nn.Module):
    """Context (stride-subsampled steps of the previous attempt: state, action, per-step feedback) -> transformer -> cross-attention decoder."""

    def __init__(self, a_dim, Tc, args):
        super().__init__()
        d, H = args.d_model, args.n_heads
        self.Tc = Tc
        for n, dim in (("s", S_DIM), ("y", Y_DIM)):
            self.register_buffer(f"{n}_mean", torch.zeros(dim))
            self.register_buffer(f"{n}_std", torch.ones(dim))
        self.tok = nn.Linear(S_DIM + a_dim + Y_DIM, d)
        self.pos = nn.Embedding(Tc, d)
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, H, 4 * d, dropout=0.0, batch_first=True, norm_first=True), args.n_layers, enable_nested_tensor=False)
        self.q_in = nn.Sequential(nn.Linear(S_DIM, d), nn.ReLU(), nn.Linear(d, d))
        self.cross = nn.ModuleList([nn.MultiheadAttention(d, H, batch_first=True) for _ in range(args.n_layers)])
        self.ln_q = nn.ModuleList([nn.LayerNorm(d) for _ in range(args.n_layers)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d)) for _ in range(args.n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, a_dim))

    def set_norm(self, s, y):
        s, y = s.reshape(-1, S_DIM), y.reshape(-1, Y_DIM)
        self.s_mean.copy_(s.mean(0)), self.s_std.copy_(s.std(0).clamp_min(1e-3))
        self.s_mean[42], self.s_std[42] = 0.0, 1.0  # mask flag stays 0/1
        if len(y) > 1:  # bc has no feedback
            self.y_mean.copy_(y.mean(0)), self.y_std.copy_(y.std(0).clamp_min(1e-3))

    def encode(self, cs, ca, cy, valid):
        """cs (B,Tc,S) raw, ca (B,Tc,A), cy (B,Tc,Y) raw, valid (B,Tc) bool (all False = no context). -> memory, key padding mask."""
        B = cs.shape[0]
        x = torch.cat([(cs - self.s_mean) / self.s_std, ca, (cy - self.y_mean) / self.y_std], -1)
        tok = torch.cat([self.null.expand(B, 1, -1), self.tok(x) + self.pos.weight[None]], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=cs.device), ~valid], 1)
        return self.enc(tok, src_key_padding_mask=pad), pad

    def decode(self, s, memory, pad):
        q = self.q_in((s - self.s_mean) / self.s_std)
        for attn, ln, mlp in zip(self.cross, self.ln_q, self.mlp):
            q = q + attn(ln(q), memory, memory, key_padding_mask=pad, need_weights=False)[0]
            q = q + mlp(q)
        return self.head(q)


def ctx_index(T, stride, device):
    idx = torch.arange(stride - 1, T, stride, device=device)
    return idx if idx[-1] == T - 1 else torch.cat([idx, torch.tensor([T - 1], device=device)])


def empty_ctx(N, Tc, A, device):
    return dict(s=torch.zeros(N, Tc, S_DIM, device=device), a=torch.zeros(N, Tc, A, device=device),
                y=torch.zeros(N, Tc, Y_DIM, device=device), valid=torch.zeros(N, Tc, dtype=torch.bool, device=device))


def make_ctx(ep, stride):
    idx = ctx_index(ep["obs"].shape[1], stride, ep["obs"].device)
    return dict(s=ep["obs"][:, idx], a=ep["act"][:, idx], y=ep["y"][:, idx], valid=ep["alive"][:, idx])


def rollout(venv, expert, policy, args, goals, seed, mode, ctx=None, ctx_mode="normal", fake=None, mask_pose=False):
    """One attempt on every task. mode: expert (true pose) | wrong_expert (expert acts on the pose of `fake`) | student (conditioned on `ctx`).
    Stored obs always contain the true hole pose (flag 0); labels = expert actions on the true pose; y = per-step feedback with the true pose."""
    device, N, T = venv.device, args.num_envs, args.num_steps
    low, high = (torch.from_numpy(b).to(device) for b in (venv.single_action_space.low, venv.single_action_space.high))
    obs, _ = venv.reset(seed=seed, options=TASK.reset_options(goals))
    if mode == "student":
        cs, ca, cy, cv = ctx["s"], ctx["a"], ctx["y"], ctx["valid"]
        if ctx_mode == "shuf_y":
            cy = cy[torch.randperm(N, device=device)]
        elif ctx_mode == "no_y":
            cy = torch.zeros_like(cy)
        elif ctx_mode == "no_ctx":
            cv = torch.zeros_like(cv)
        if mask_pose:
            cs = apply_mask(cs, torch.ones(N, dtype=torch.bool, device=device))
        with torch.no_grad():
            memory, pad = policy.encode(cs, ca, cy, cv)
    xs, acts, labels, alive, ys = [], [], [], [], []
    done = torch.zeros(N, dtype=torch.bool, device=device)
    for t in range(T):
        with torch.no_grad():
            a_lab = expert.get_action(obs[..., :43], deterministic=True)
            if mode == "expert":
                a = a_lab
            elif mode == "wrong_expert":
                a = expert.get_action(wrong_expert_obs(obs, goals, fake), deterministic=True)
            else:
                a = policy.decode(sobs(obs, mask_pose)[:, None], memory, pad)[:, 0]
        a, a_lab = torch.max(torch.min(a, high), low), torch.max(torch.min(a_lab, high), low)
        xs.append(sobs(obs)), acts.append(a), labels.append(a_lab), alive.append(~done), ys.append(step_feedback(obs))
        obs, _, _, _, infos = venv.step(a)
        done |= (infos["final_info"] if "final_info" in infos else infos)["success"].bool()  # terminate at the first success
    return dict(obs=torch.stack(xs, 1), act=torch.stack(acts, 1), lab=torch.stack(labels, 1), alive=torch.stack(alive, 1),
                y=torch.stack(ys, 1), succ=done)


def train(policy, opt, sched, D, args, icl):
    """grad steps on the aggregated data. icl: D has Q (M,2,T,S), L, A (M,2,T), ctx s/a/y/valid (M,Tc,...); bc: Q (M,T,S), L, A."""
    device = D["Q"].device
    M, B = D["Q"].shape[0], args.batch_size
    losses = []
    for _ in range(args.grad_steps):
        idx = torch.randint(0, M, (B,), device=device)
        if icl:
            j = torch.randint(0, 2, (B,), device=device)
            h = torch.where(j == 1, torch.randint(0, 2, (B,), device=device), torch.zeros_like(j))  # attempt 1: h=0; attempt 2: h in {0, 1}
            q, lab, m = D["Q"][idx, j], D["L"][idx, j], D["A"][idx, j].float()
            drop = torch.rand(B, device=device) < args.pose_dropout
            cs, ca, cy = apply_mask(D["cs"][idx], drop), D["ca"][idx], D["cy"][idx]
            valid = D["cv"][idx] & (h == 1)[:, None]
            q = apply_mask(q, drop)
        else:
            q, lab, m = D["Q"][idx], D["L"][idx], D["A"][idx].float()
            cs, ca, cy, valid = D["z"]["s"][:B], D["z"]["a"][:B], D["z"]["y"][:B], D["z"]["valid"][:B]
        memory, pad = policy.encode(cs, ca, cy, valid)
        pred = policy.decode(q, memory, pad)
        loss = (((pred - lab) ** 2).mean(-1) * m).sum() / m.sum().clamp_min(1.0)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
    return sum(losses[-100:]) / 100


def main():
    args = tyro.cli(Args)
    assert args.mode in ("icl", "bc")
    icl = args.mode == "icl"
    device = torch.device("cuda")
    run = RunDir(args.out_dir or str(LOG_ROOT / "runs" / f"checker-{args.exp_name}"), args)
    torch.manual_seed(args.seed)
    env = gym.make("PegInsertionSideBias-v1", num_envs=args.num_envs, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    venv = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True, record_metrics=True)
    A = venv.single_action_space.shape[0]

    class _Space:
        single_observation_space = type("S", (), {"shape": (43,)})
        single_action_space = venv.single_action_space

    expert = Agent(_Space).to(device)
    expert.load_state_dict(torch.load(args.expert_checkpoint, map_location=device))
    expert.eval()
    Tc = len(ctx_index(args.num_steps, args.ctx_stride, device))
    policy = Policy(A, Tc, args).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_iterations * args.grad_steps, eta_min=args.lr * 0.02)
    N = args.num_envs

    gens = {r: torch.Generator(device=device).manual_seed(args.eval_seed + 1000 * (i + 1)) for i, r in enumerate(("white", "black"))}
    eval_goals = {r: checker_sample(N, r, device, gens[r], args.split) for r in gens}
    eval_fake = {r: checker_sample(N, "all", device, gens[r], args.split) for r in gens}

    def evaluate(ctx_mode="normal", region_list=("white", "black")):
        """attempt 1 (no context), attempt 2 (conditioned on attempt 1 unless ctx_mode == no_ctx, which is also what BC does)."""
        policy.eval()
        out = {}
        for r in region_list:
            g = eval_goals[r]
            e1 = rollout(venv, expert, policy, args, g, args.eval_seed, "student", empty_ctx(N, Tc, A, device))
            e2 = rollout(venv, expert, policy, args, g, args.eval_seed + 1, "student", make_ctx(e1, args.ctx_stride),
                         "no_ctx" if (not icl and ctx_mode == "normal") else ctx_mode)
            out[r] = dict(pass1=e1["succ"].float().mean().item(), pass2=(e1["succ"] | e2["succ"]).float().mean().item(),
                          attempt2=e2["succ"].float().mean().item(), len1=e1["alive"].sum(1).float().mean().item(), len2=e2["alive"].sum(1).float().mean().item())
        policy.train()
        return out

    refs = {}
    for r in ("white", "black"):
        e = rollout(venv, expert, policy, args, eval_goals[r], args.eval_seed, "expert")
        w = rollout(venv, expert, policy, args, eval_goals[r], args.eval_seed, "wrong_expert", fake=eval_fake[r])
        refs[r] = dict(expert=e["succ"].float().mean().item(), wrong_expert=w["succ"].float().mean().item())
        print(f"reference {r}: expert (true pose) {refs[r]['expert']:.3f}, expert on a random wrong pose {refs[r]['wrong_expert']:.3f}", flush=True)
    if args.eval_envs_only:
        return

    # fixed validation sets: clean expert demos (true pose) on the eval goals of each region; val loss = action MSE of the policy (no context) vs the expert labels
    val = {}
    for r in ("white", "black"):
        e = rollout(venv, expert, policy, args, eval_goals[r], args.eval_seed + 7, "expert")
        val[r] = (e["obs"], e["lab"], e["alive"].float())

    def val_loss():
        policy.eval()
        out = {}
        with torch.no_grad():
            for r, (o, lab, m) in val.items():
                tot = 0.0
                for i in range(0, N, 128):  # chunks of tasks
                    c = empty_ctx(len(o[i:i + 128]), Tc, A, device)
                    memory, pad = policy.encode(c["s"], c["a"], c["y"], c["valid"])
                    pred = policy.decode(o[i:i + 128], memory, pad)
                    tot += (((pred - lab[i:i + 128]) ** 2).mean(-1) * m[i:i + 128]).sum().item()
                out[r] = tot / m.sum().item()
        policy.train()
        return out

    keys = ("Q", "L", "A") + (("cs", "ca", "cy", "cv") if icl else ())
    D = {k: [] for k in keys}
    log = []
    for it in range(args.num_iterations):
        t0 = time.time()
        seed = args.seed * 1000 + it * 10
        info = {}
        if icl:
            goals, fake = checker_sample(N, "white", device, split=args.split), checker_sample(N, "all", device, split=args.split)
            e1 = rollout(venv, expert, policy, args, goals, seed, "wrong_expert", fake=fake)
            c = make_ctx(e1, args.ctx_stride)
            e2 = rollout(venv, expert, policy, args, goals, seed + 1, "student", c, mask_pose=True)
            for k, v in zip(keys, (torch.stack([e1["obs"], e2["obs"]], 1), torch.stack([e1["lab"], e2["lab"]], 1), torch.stack([e1["alive"], e2["alive"]], 1),
                                   c["s"], c["a"], c["y"], c["valid"])):
                D[k].append(v)
            info = dict(wrong_expert_success=e1["succ"].float().mean().item(), student_masked_success=e2["succ"].float().mean().item())
        else:
            for e in range(2):  # 2N clean demos on white goals
                ep = rollout(venv, expert, policy, args, checker_sample(N, "white", device, split=args.split), seed + e, "expert")
                for k, v in zip(keys, (ep["obs"], ep["lab"], ep["alive"])):
                    D[k].append(v)
        Dc = {k: torch.cat(v) for k, v in D.items()}
        if not icl:
            Dc["z"] = empty_ctx(args.batch_size, Tc, A, device)
        if it == 0:
            policy.set_norm(Dc["Q"], D["cy"][0] if icl else torch.zeros(1, Y_DIM, device=device))
        loss = train(policy, opt, sched, Dc, args, icl)
        ev = evaluate()
        vl = val_loss()
        n_traj = Dc["Q"].shape[0] * (2 if icl else 1)
        row = dict(iteration=it, loss=loss, val_loss=vl, n_traj=n_traj, eval=ev, seconds=time.time() - t0, **info)
        log.append(row)
        print(f"it {it:2d} loss {loss:.4f} val white {vl['white']:.4f} black {vl['black']:.4f} traj {n_traj} | " + " | ".join(
            f"{r}: pass@1 {v['pass1']:.3f} pass@2 {v['pass2']:.3f}" for r, v in ev.items()) + f" | {info} {row['seconds']:.0f}s", flush=True)
        run.log(it, dict(loss=loss, val_loss_white=vl['white'], val_loss_black=vl['black'], **{f"eval/{r}_{k}": x for r, v in ev.items() for k, x in v.items()}, **info))
        torch.save(policy.state_dict(), run.ckpt("policy_latest.pt"))

    controls = {"normal": log[-1]["eval"]}
    if icl:
        for cm in ("no_y", "shuf_y", "no_ctx"):
            controls[cm] = evaluate(cm)
            print(f"control {cm:7s} " + " | ".join(f"{r}: pass@1 {v['pass1']:.3f} pass@2 {v['pass2']:.3f}" for r, v in controls[cm].items()), flush=True)
    torch.save(policy.state_dict(), run.ckpt("policy_final.pt"))
    run.save_json("results.json", dict(mode=args.mode, refs=refs, controls=controls, final_eval=log[-1]["eval"], log=log))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, r in zip(ax, ("white", "black")):
        for key in ("pass1", "pass2"):
            a.plot([row["eval"][r][key] for row in log], label=key)
        a.axhline(refs[r]["expert"], color="k", ls="--", label="expert")
        a.set_title(f"{args.mode} on {r}")
        a.set_xlabel("iteration")
        a.set_ylim(0, 1)
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(run.plot("curve.png"), dpi=120)


if __name__ == "__main__":
    main()
