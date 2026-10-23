"""Evaluate a state-based PushT-v1 PPO policy when the observed pose of the T is corrupted.

Only the observation is perturbed: the policy sees the T at (true pose + offset), while the simulator keeps
the real T where it is. The offset (dx, dy, dtheta) is sampled uniformly per episode at reset (or
resampled every step with --resample-every-step). Only the T's `obj_pose` entry is touched, not the goal or robot.

Runs the same initial states at several noise scales (0 = clean baseline, 1 = the full --noise-xy/--noise-theta-deg),
prints success rates, and writes top-down overlay videos (true T vs. the T the policy believes) for the first
--num-viz-envs episodes.

Example:
    python eval_obs_noise.py --checkpoint runs/pusht-ppo-state/final_ckpt.pt
"""
import os
from dataclasses import dataclass
from typing import List

import gymnasium as gym
import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from matplotlib.patches import Polygon

import mani_skill.envs  # noqa: F401  (registers PushT-v1)
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from ppo import Agent


@dataclass
class Args:
    checkpoint: str
    """path to the trained PPO checkpoint (.pt)"""
    env_id: str = "PushT-v1"
    num_envs: int = 64
    """number of parallel episodes used for the success statistics"""
    num_steps: int = 100
    """episode length"""
    noise_xy: float = 0.05
    """max |dx|, |dy| of the observation offset in meters (sampled uniformly in [-noise_xy, noise_xy])"""
    noise_theta_deg: float = 45.0
    """max |dtheta| of the observation offset in degrees (sampled uniformly in [-theta, theta])"""
    noise_scales: List[float] = (0.0, 0.25, 0.5, 1.0)
    """multipliers on (noise_xy, noise_theta_deg) to evaluate; 0.0 is the clean baseline"""
    resample_every_step: bool = False
    """if set, resample the offset each step instead of once per episode"""
    seed: int = 1
    num_viz_envs: int = 8
    """how many episodes (the first N) to render in the overlay videos; 0 disables videos"""
    viz_scales: List[float] = (0.0, 1.0)
    """which entries of noise_scales get a video"""
    out_dir: str = ""
    """where to write videos/images (default: <checkpoint dir>/obs_noise)"""


def quat_mul(a, b):
    """Hamilton product of quaternions in (w, x, y, z) order, batched."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def perturb_obs(obs, offsets):
    """Shift the T pose (last 7 entries of the state obs: xyz, quat wxyz) by offsets=(dx, dy, dtheta)."""
    obs = obs.clone()
    pose = obs[:, -7:]
    pose[:, 0] += offsets[:, 0]
    pose[:, 1] += offsets[:, 1]
    half = offsets[:, 2] / 2
    dq = torch.stack([half.cos(), torch.zeros_like(half), torch.zeros_like(half), half.sin()], dim=-1)
    q = quat_mul(dq, pose[:, 3:7])
    pose[:, 3:7] = q / q.norm(dim=-1, keepdim=True)
    return obs


def sample_offsets(n, scale, args, device, gen):
    u = (torch.rand(n, 3, device=device, generator=gen) * 2 - 1)
    return u * torch.tensor([args.noise_xy, args.noise_xy, np.deg2rad(args.noise_theta_deg)], device=device) * scale


def yaw_of(raw_pose):
    """z-rotation angle from a (.., 7) raw pose [x y z qw qx qy qz]."""
    w, x, y, z = raw_pose[..., 3], raw_pose[..., 4], raw_pose[..., 5], raw_pose[..., 6]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# The T in its center-of-mass frame (from PushTEnv.create_tee): a horizontal bar and a vertical stem.
T_RECTS = [  # (x_min, x_max, y_min, y_max)
    (-0.1, 0.1, -0.0625, -0.0125),
    (-0.025, 0.025, -0.0125, 0.1375),
]


def tee_polys(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    out = []
    for x0, x1, y0, y1 in T_RECTS:
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        out.append(pts @ np.array([[c, s], [-s, c]]) + np.array([x, y]))
    return out


def render_video(rec, path, title, fps=15):
    """rec: dict of arrays [T, E, ...] with true/perceived/goal tee (x, y, yaw) and tcp xy."""
    T, E = rec["true"].shape[:2]
    allxy = np.concatenate([rec["true"][..., :2], rec["perc"][..., :2], rec["goal"][..., :2], rec["tcp"]], axis=1)
    lo, hi = allxy.reshape(-1, 2).min(0) - 0.25, allxy.reshape(-1, 2).max(0) + 0.25
    cols = min(4, E)
    rows = int(np.ceil(E / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows + 0.4), squeeze=False)
    frames = []
    for t in range(T):
        for e, ax in enumerate(axs.ravel()[:E]):
            ax.clear()
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for p in tee_polys(*rec["goal"][t, e]):
                ax.add_patch(Polygon(p, closed=True, fc="#c9ced4", ec="none"))
            for p in tee_polys(*rec["true"][t, e]):
                ax.add_patch(Polygon(p, closed=True, fc="#d2373b", ec="none", alpha=0.85))
            for p in tee_polys(*rec["perc"][t, e]):
                ax.add_patch(Polygon(p, closed=True, fc="none", ec="#2a6fdb", lw=1.8, ls="--"))
            ax.plot(*rec["tcp"][t, e], "o", color="#1f2328", ms=5)
            ok = rec["success"][: t + 1, e].any()
            ax.set_title(f"env {e}  {'SUCCESS' if ok else ''}", fontsize=9, color="#1a7f37" if ok else "#1f2328")
        for ax in axs.ravel()[E:]:
            ax.axis("off")
        fig.suptitle(f"{title} | step {t}   red = true T, blue dashed = T the policy sees, grey = goal", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    imageio.mimsave(path, frames, fps=fps)
    return frames


def main():
    args = tyro.cli(Args)
    device = torch.device("cuda")
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "obs_noise")
    os.makedirs(out_dir, exist_ok=True)

    env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="state", control_mode="pd_joint_delta_pos",
                   sim_backend="physx_cuda", reconfiguration_freq=1)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    base = env.unwrapped
    venv = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True, record_metrics=True)

    agent = Agent(venv).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    results = {}
    for scale in args.noise_scales:
        gen = torch.Generator(device=device).manual_seed(args.seed)
        obs, _ = venv.reset(seed=args.seed)
        offsets = sample_offsets(args.num_envs, scale, args, device, gen)
        # sanity check: the last 7 obs entries really are the simulator's T pose
        assert torch.allclose(obs[:, -7:], base.tee.pose.raw_pose, atol=1e-4), "obs layout assumption broken"

        succ_once = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
        succ_end = torch.zeros_like(succ_once)
        rec = {k: [] for k in ("true", "perc", "goal", "tcp", "success")}
        nv = min(args.num_viz_envs, args.num_envs)
        for _ in range(args.num_steps):
            if args.resample_every_step:
                offsets = sample_offsets(args.num_envs, scale, args, device, gen)
            seen = perturb_obs(obs, offsets)
            with torch.no_grad():
                action = agent.get_action(seen, deterministic=True)
            if nv:
                true_pose = base.tee.pose.raw_pose[:nv]
                rec["true"].append(torch.cat([true_pose[:, :2], yaw_of(true_pose)[:, None]], -1).cpu().numpy())
                rec["perc"].append(torch.cat([seen[:nv, -7:-5], yaw_of(seen[:nv, -7:])[:, None]], -1).cpu().numpy())
                g = base.goal_tee.pose.raw_pose[:nv]
                rec["goal"].append(torch.cat([g[:, :2], yaw_of(g)[:, None]], -1).cpu().numpy())
                rec["tcp"].append(base.agent.tcp.pose.p[:nv, :2].cpu().numpy())
            obs, _, _, _, infos = venv.step(action)
            # on the last step the vector env auto-resets, so the real success flag is in final_info
            s = (infos["final_info"] if "final_info" in infos else infos)["success"].bool()
            succ_once |= s
            succ_end = s
            if nv:
                rec["success"].append(s[:nv].cpu().numpy())
        results[scale] = (succ_once.float().mean().item(), succ_end.float().mean().item())
        print(f"scale={scale:<5} xy=±{args.noise_xy*scale*100:.1f}cm theta=±{args.noise_theta_deg*scale:.0f}deg  "
              f"success_once={results[scale][0]:.3f}  success_at_end={results[scale][1]:.3f}", flush=True)

        if nv and scale in args.viz_scales:
            rec = {k: np.stack(v) for k, v in rec.items()}
            path = os.path.join(out_dir, f"overlay_scale{scale}.mp4")
            frames = render_video(rec, path, f"obs noise scale {scale}")
            imageio.imwrite(os.path.join(out_dir, f"overlay_scale{scale}_final.png"), frames[-1])
            imageio.imwrite(os.path.join(out_dir, f"overlay_scale{scale}_step0.png"), frames[0])
            print(f"  wrote {path}", flush=True)

    venv.close()


if __name__ == "__main__":
    main()
