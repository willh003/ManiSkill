"""On-policy distillation with reward access: train a recurrent student that only sees a noisy T pose.

The student (GRU) acts from the noisy observation `[noisy_obs_t, a_{t-1}, r_{t-1}]`, so it is conditioned on the
history of states, actions and rewards. Each iteration (DAgger-style):
  1. roll out the student for k steps under the noisy observation, then let the expert PPO policy (which sees the
     true state) act for the rest of the episode; k is drawn per episode from a curriculum that grows over iterations
  2. aggregate the (input, executed action) sequences, plus a mask marking the steps t >= k where the expert acted
  3. retrain the student to predict the expert's action a_t from its history; the loss is only taken on the expert
     steps (t >= k), so the student-controlled prefix gives no supervision, but it is part of the history the
     student conditions on
  4. evaluate the student alone under noise.

The T-pose noise is a per-episode offset (dx, dy, dtheta) applied only to the observation (see eval_obs_noise.py).

Example:
    python distill_recurrent.py --exp-name distill
"""
import faulthandler

# some runs hang at startup; dump every thread's stack to the log after 5 minutes (cancelled once startup is done)
faulthandler.dump_traceback_later(300, exit=False)

import json
import os
import time
from dataclasses import dataclass

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tyro

import mani_skill.envs  # noqa: F401  (registers PushT-v1)
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from eval_obs_noise import perturb_obs, render_video, sample_offsets, yaw_of
from ppo import Agent

# (lo, hi) fraction of the episode the student acts before the expert takes over, per iteration.
# Iterations beyond the list reuse the last range.
CURRICULUM = [(0.2, 0.5), (0.3, 0.8), (0.4, 0.9)]


@dataclass
class Args:
    expert_checkpoint: str = "runs/pusht-ppo-state/final_ckpt.pt"
    exp_name: str = "distill"
    out_dir: str = ""
    """default: runs/pusht-distill-<exp_name>"""
    env_id: str = "PushT-v1"
    num_envs: int = 256
    """episodes per iteration (and per evaluation)"""
    num_steps: int = 100
    """episode length"""
    num_iterations: int = 5
    noise_xy: float = 0.05
    """max |dx|, |dy| of the observation offset in meters"""
    noise_theta_deg: float = 45.0
    """max |dtheta| of the observation offset in degrees"""
    noise_scale: float = 1.0
    """multiplier on the observation offset; 0 gives the student the true (noise-free) obs as a sanity check"""
    hidden: int = 256
    grad_steps: int = 1000
    """gradient steps per iteration, batches sampled from the aggregated dataset"""
    batch_size: int = 128
    """sequences per gradient step"""
    lr: float = 5e-4
    k_lo_frac: float = -1.0
    """if >= 0, overrides the lower end of every CURRICULUM range (0 lets the handoff happen as early as step 1, so the
    first steps also get supervision); the upper end still follows the curriculum"""
    relabel: bool = False
    """if set, the target at every step (including the student-controlled prefix) is the expert's action on the true
    state and the loss is taken on all steps (DAgger); by default the loss is only on the expert steps t >= k"""
    memoryless: bool = False
    """ablation: the student is a feed-forward policy a|s on the noisy observation only (no history, previous action or
    reward, no recurrence); same depth and width otherwise"""
    no_reward: bool = False
    """ablation: the reward input is always 0, so the student only sees states and its own past actions"""
    expert_in_history: bool = False
    """ablation: the student input also contains the expert's action at the previous state (the expert acting on the
    true state, a privileged signal), i.e. history entries (s, a, a*, r). The evaluation computes it the same way"""
    lr_decay: bool = False
    """cosine-decay the learning rate to 2% of --lr over all gradient steps"""
    explore_std: float = 0.0
    """gaussian noise added to the student's actions while it is acting during data collection"""
    seed: int = 1
    eval_seed: int = 12345
    num_viz_envs: int = 8
    """episodes rendered in the final overlay video (0 disables it)"""


class Student(nn.Module):
    """GRU policy over inputs [obs, prev_action, (prev_expert_action,) prev_reward]."""

    def __init__(self, in_dim, act_dim, hidden, memoryless=False):
        super().__init__()
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std", torch.ones(in_dim))
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        # memoryless: a plain layer in place of the GRU, so depth/width match and nothing carries over between steps
        self.gru = None if memoryless else nn.GRU(hidden, hidden, batch_first=True)
        self.mid = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU()) if memoryless else None
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, act_dim))

    def set_norm(self, x):
        flat = x.reshape(-1, x.shape[-1])
        std = flat.std(0)
        self.in_mean.copy_(flat.mean(0))
        self.in_std.copy_(torch.where(std < 1e-4, torch.ones_like(std), std))  # constant dims (e.g. goal_pos)

    def forward(self, x, h=None):
        z = self.enc((x - self.in_mean) / self.in_std)
        if self.gru is None:
            return self.head(self.mid(z)), None
        y, h = self.gru(z, h)
        return self.head(y), h


def run_episodes(venv, base, expert, student, args, seed, k, explore_std=0.0, rec_n=0, expert_obs="true"):
    """Roll out `num_envs` episodes. Env i is driven by the student while t < k[i], by the expert afterwards.

    expert_obs: "true" (expert sees the real state) or "noisy" (expert sees the perturbed one; used for baselines).
    Returns inputs, targets and the expert-step mask for training plus success metrics (and a recording of the first rec_n envs).
    """
    device, N, T = venv.device, args.num_envs, args.num_steps
    low = torch.from_numpy(venv.single_action_space.low).to(device)
    high = torch.from_numpy(venv.single_action_space.high).to(device)
    A = low.shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    obs, _ = venv.reset(seed=seed)
    offsets = sample_offsets(N, args.noise_scale, args, device, gen)
    prev_a, prev_r, h = torch.zeros(N, A, device=device), torch.zeros(N, 1, device=device), None
    prev_aexp = torch.zeros(N, A, device=device)  # expert action at the previous state (only used with --expert-in-history)
    succ_once = torch.zeros(N, dtype=torch.bool, device=device)
    xs, ys, ms, rec = [], [], [], {key: [] for key in ("true", "perc", "goal", "tcp", "success")}
    act_mse = 0.0
    for t in range(T):
        noisy = perturb_obs(obs, offsets)
        if args.memoryless:
            x = noisy
        else:
            x = torch.cat([noisy, prev_a] + ([prev_aexp] if args.expert_in_history else []) + [prev_r], dim=-1)
        with torch.no_grad():
            a_exp = expert.get_action(noisy if expert_obs == "noisy" else obs, deterministic=True)
            a_stu, h = student(x[:, None], h)
            a_stu = a_stu[:, 0]
            if explore_std > 0:
                a_stu = a_stu + torch.randn_like(a_stu) * explore_std
        a_exp, a_stu = torch.max(torch.min(a_exp, high), low), torch.max(torch.min(a_stu, high), low)
        a_exec = torch.where((t < k)[:, None], a_stu, a_exp)
        act_mse += ((a_stu - a_exp) ** 2).mean().item() / T
        xs.append(x)
        ys.append(a_exp if args.relabel else a_exec)
        # True where the loss is taken: the expert-controlled steps (all steps when relabeling)
        ms.append(torch.ones_like(k, dtype=torch.bool) if args.relabel else t >= k)
        if rec_n:
            true_pose = base.tee.pose.raw_pose[:rec_n]
            goal = base.goal_tee.pose.raw_pose[:rec_n]
            rec["true"].append(torch.cat([true_pose[:, :2], yaw_of(true_pose)[:, None]], -1).cpu().numpy())
            rec["perc"].append(torch.cat([noisy[:rec_n, -7:-5], yaw_of(noisy[:rec_n, -7:])[:, None]], -1).cpu().numpy())
            rec["goal"].append(torch.cat([goal[:, :2], yaw_of(goal)[:, None]], -1).cpu().numpy())
            rec["tcp"].append(base.agent.tcp.pose.p[:rec_n, :2].cpu().numpy())
        obs, r, _, _, infos = venv.step(a_exec)
        # on the last step the vector env auto-resets, so the real success flag is in final_info
        s = (infos["final_info"] if "final_info" in infos else infos)["success"].bool()
        succ_once |= s
        if rec_n:
            rec["success"].append(s[:rec_n].cpu().numpy())
        prev_a, prev_aexp = a_exec, a_exp
        prev_r = torch.zeros_like(prev_r) if args.no_reward else r[:, None].float()
    out = dict(
        x=torch.stack(xs, 1), y=torch.stack(ys, 1), mask=torch.stack(ms, 1).float(),
        success_once=succ_once.float().mean().item(), success_at_end=s.float().mean().item(), act_mse=act_mse,
        succ_once_env=succ_once.cpu(),
    )
    if rec_n:
        out["rec"] = {key: np.stack(v) for key, v in rec.items()}
    return out


def sample_k(n, it, args, device, gen):
    lo, hi = CURRICULUM[min(it, len(CURRICULUM) - 1)]
    if args.k_lo_frac >= 0:
        lo = args.k_lo_frac
    f = lo + (hi - lo) * torch.rand(n, device=device, generator=gen)
    return (f * args.num_steps).round().clamp(1, args.num_steps - 1).long()


def main():
    args = tyro.cli(Args)
    device = torch.device("cuda")
    out_dir = args.out_dir or f"runs/pusht-distill-{args.exp_name}"
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="state", control_mode="pd_joint_delta_pos",
                   sim_backend="physx_cuda")
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    base = env.unwrapped
    venv = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True, record_metrics=True)
    obs_dim, act_dim = venv.single_observation_space.shape[0], venv.single_action_space.shape[0]

    expert = Agent(venv).to(device)
    expert.load_state_dict(torch.load(args.expert_checkpoint, map_location=device))
    expert.eval()
    in_dim = obs_dim if args.memoryless else obs_dim + act_dim + (act_dim if args.expert_in_history else 0) + 1
    student = Student(in_dim, act_dim, args.hidden, args.memoryless).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.num_iterations * args.grad_steps, eta_min=args.lr * 0.02) if args.lr_decay else None

    # sanity check: the last 7 obs entries are the simulator's T pose (the entries we perturb)
    o, _ = venv.reset(seed=args.seed)
    assert torch.allclose(o[:, -7:], base.tee.pose.raw_pose, atol=1e-4), "obs layout assumption broken"

    never = torch.full((args.num_envs,), args.num_steps, device=device)  # k = H: student acts the whole episode
    always_expert = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    baselines = {}
    for name, kind in (("expert_clean", "true"), ("expert_noisy", "noisy")):
        r = run_episodes(venv, base, expert, student, args, args.eval_seed, always_expert, expert_obs=kind)
        baselines[name] = (r["success_once"], r["success_at_end"])
        print(f"baseline {name:13s} success_once={r['success_once']:.3f} success_at_end={r['success_at_end']:.3f}", flush=True)

    faulthandler.cancel_dump_traceback_later()
    gen = torch.Generator(device=device).manual_seed(args.seed)
    data_x, data_y, data_m, log = [], [], [], []
    for it in range(args.num_iterations):
        t0 = time.time()
        k = sample_k(args.num_envs, it, args, device, gen)
        r = run_episodes(venv, base, expert, student, args, args.seed + 1 + it, k, args.explore_std)
        data_x.append(r["x"])
        data_y.append(r["y"])
        data_m.append(r["mask"])
        X, Y, M = torch.cat(data_x), torch.cat(data_y), torch.cat(data_m)
        if it == 0:
            student.set_norm(X)
        losses = []
        for _ in range(args.grad_steps):
            idx = torch.randint(0, X.shape[0], (args.batch_size,), device=device)
            pred, _ = student(X[idx])
            m = M[idx]
            # supervise only the expert-controlled steps (t >= k); the student prefix is input only
            loss = (((pred - Y[idx]) ** 2).mean(-1) * m).sum() / m.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            if sched:
                sched.step()
            losses.append(loss.item())
        student.eval()
        ev = run_episodes(venv, base, expert, student, args, args.eval_seed, never)
        student.train()
        row = dict(iteration=it + 1, k_range=CURRICULUM[min(it, len(CURRICULUM) - 1)], dataset_episodes=X.shape[0],
                   loss=float(np.mean(losses[-50:])), student_success_once=ev["success_once"],
                   student_success_at_end=ev["success_at_end"], student_vs_expert_action_mse=ev["act_mse"],
                   rollout_success_once=r["success_once"])
        log.append(row)
        print(json.dumps(row) + f"  ({time.time() - t0:.0f}s)", flush=True)
        torch.save(student.state_dict(), os.path.join(out_dir, "student_latest.pt"))

    json.dump(dict(args=vars(args), baselines=baselines, log=log), open(os.path.join(out_dir, "log.json"), "w"), indent=1)

    its = [row["iteration"] for row in log]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(its, [row["student_success_once"] for row in log], marker="o", color="#2a6fdb", label="student success_once")
    ax.plot(its, [row["student_success_at_end"] for row in log], marker="s", color="#d9701f", label="student success_at_end")
    ax.axhline(baselines["expert_clean"][0], color="#656d76", ls=":", label="expert, true obs")
    ax.axhline(baselines["expert_noisy"][0], color="#656d76", ls="--", label="expert, noisy obs")
    ax.set_xlabel("DAgger iteration"); ax.set_ylabel("success under noise"); ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(its); ax.grid(color="#e4e7eb"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "success_curve.png"), dpi=150)

    if args.num_viz_envs:
        student.eval()
        ev = run_episodes(venv, base, expert, student, args, args.eval_seed, never, rec_n=args.num_viz_envs)
        path = os.path.join(out_dir, "student_overlay.mp4")
        frames = render_video(ev["rec"], path, "student only, noisy obs")
        import imageio
        imageio.imwrite(os.path.join(out_dir, "student_overlay_final.png"), frames[-1])
        print(f"wrote {path}", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
