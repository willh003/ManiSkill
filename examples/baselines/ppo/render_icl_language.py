"""Video of a language-feedback peg student: for tasks where the h=0 episode fails and the h=1 episode (conditioned on the h=0 episode and its
language command) succeeds, show h=0 on the top row and h=1 on the bottom row, with the command under each h=1 clip.

Pass 1 (no rendering) rolls out h=0, h=1 on all eval tasks (same as the evaluation in icl_episodic.py) and picks the tasks; pass 2 repeats the
same seeds and renders only the picked envs.

usage: python render_icl_language.py --run peg-language-hidden-strong [--out peg_language.mp4] [--n 4]
"""
import json
import textwrap
from dataclasses import dataclass

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw, ImageFont

import icl_episodic as ie
from icl_tasks import TASKS, command_text
from ppo import Agent
from run_logging import LOG_ROOT
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


@dataclass
class Cfg:
    run: str = "peg-language-hidden-strong"
    """run name without the icl- prefix (logs/runs/icl-<run>/)"""
    expert: str = str(LOG_ROOT / "runs/peg-expert-r1/ckpt_3576.pt")
    out: str = "peg_language.mp4"
    """written to logs/renders/icl-peg/videos/"""
    n: int = 4
    w: int = 480
    h: int = 360
    fps: int = 15


def text_bar(w, h, lines, color, size=15):
    img = Image.new("RGB", (w, h), (24, 24, 27))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    for i, (t, c) in enumerate(lines):
        d.text((8, 4 + i * (size + 3)), t, fill=c, font=font)
    return np.asarray(img)


def main():
    cfg = tyro.cli(Cfg)
    run_dir = LOG_ROOT / "runs" / f"icl-{cfg.run}"
    raw = json.load(open(run_dir / "config.json"))["args"]
    args = ie.Args(**{k: v for k, v in raw.items() if k in ie.Args.__dataclass_fields__})
    assert args.feedback == "language"
    ie.TASK = TASKS[args.env_id](hide_hole=args.hide_hole, full_info=args.full_info)
    TASK, device = ie.TASK, torch.device("cuda")

    env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda",
                   render_mode="rgb_array", human_render_camera_configs=dict(width=cfg.w, height=cfg.h))
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    base = env.unwrapped
    venv = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True, record_metrics=True)
    obs_dim, act_dim = venv.single_observation_space.shape[0], venv.single_action_space.shape[0]

    class _Space:
        single_observation_space = type("S", (), {"shape": (TASK.expert_obs(torch.zeros(1, obs_dim)).shape[-1],)})
        single_action_space = venv.single_action_space

    expert = Agent(_Space).to(device)
    expert.load_state_dict(torch.load(cfg.expert, map_location=device))
    expert.eval()
    s_dim, y_dim = ie.student_obs(torch.zeros(1, obs_dim)).shape[-1], ie.feedback_dim(args)
    policy = ie.ICLPolicy(s_dim, act_dim, y_dim, args).to(device)
    policy.load_state_dict(torch.load(run_dir / "checkpoints" / "policy_final.pt", map_location=device))
    policy.eval()

    with torch.random.fork_rng():
        torch.manual_seed(args.eval_seed + 1000)
        goals = TASK.sample(base, args.num_envs)
    A = act_dim
    ctx = ie.new_ctx(args, args.num_envs, s_dim, A, y_dim, device)

    def episode(h, cb=None):
        c = dict(ctx, n=torch.full_like(ctx["n"], h))
        return ie.rollout(venv, expert, policy, args, goals, args.eval_seed + h, "student", c, step_cb=cb)

    # pass 1
    ep0 = episode(0)
    ie.add_to_ctx(ctx, ep0, args)
    ep1 = episode(1)
    s0, s1 = ep0["success_once"], ep1["success_once"]
    print(f"pass 1: success_once h0 {s0.float().mean():.3f} h1 {s1.float().mean():.3f}; h0 fail & h1 success: {(~s0 & s1).sum().item()}")
    cmds = command_text(ctx["y"][:, 0].long())
    idx = (~s0 & s1).nonzero().flatten()[: 3 * cfg.n].tolist()  # candidates; pass 2 can differ slightly (GPU sim is not bit-reproducible)
    assert len(idx) >= cfg.n, "not enough fail -> success tasks"

    # pass 2: same seeds, render the picked envs
    frames = {0: [], 1: []}
    succ = {0: [], 1: []}

    def make_cb(h):
        def cb(t, succ_once):
            img = base.render_rgb_array()  # (N, H, W, 3) uint8 (or a single tiled image)
            img = img.cpu().numpy() if torch.is_tensor(img) else np.asarray(img)
            frames[h].append(img[idx])
            succ[h].append(succ_once[idx].cpu().numpy().copy())
        return cb

    ctx2 = ie.new_ctx(args, args.num_envs, s_dim, A, y_dim, device)
    ctx.update(ctx2)  # reset the context buffers in place
    ep0b = episode(0, make_cb(0))
    ie.add_to_ctx(ctx, ep0b, args)
    ep1b = episode(1, make_cb(1))
    print("pass 2 outcomes of the picked tasks: h0", ep0b["success_once"][idx].tolist(), "h1", ep1b["success_once"][idx].tolist())
    cmds2 = command_text(ctx["y"][:, 0].long())
    ok = [j for j, e in enumerate(idx) if not ep0b["success_once"][e] and ep1b["success_once"][e]][: cfg.n]
    assert len(ok) == cfg.n, ok
    idx = [idx[j] for j in ok]
    for h in (0, 1):
        frames[h] = [f[ok] for f in frames[h]]
        succ[h] = [x[ok] for x in succ[h]]
    print("tasks", idx)

    T = len(frames[0])
    W, H = cfg.w, cfg.h
    out = []
    for t in range(T):
        rows = []
        for h in (0, 1):
            cells = []
            for j, e in enumerate(idx):
                ok = succ[h][t][j]
                tag = "SUCCESS" if ok else ""
                col = (110, 220, 140) if ok else (235, 235, 235)
                if h == 0:
                    head = text_bar(W, 34, [(f"task {j} | h=0 (no feedback)  {tag}", col)], col)
                    foot = text_bar(W, 0 + 6, [], col)
                else:
                    wrapped = textwrap.wrap("Feedback: " + cmds2[e], 54)[:3]
                    head = text_bar(W, 34, [(f"task {j} | h=1 (after the command)  {tag}", col)], col)
                    foot = text_bar(W, 4 + 18 * 3, [(w_, (255, 214, 102)) for w_ in wrapped], col)
                cells.append(np.concatenate([head, frames[h][t][j], foot], 0))
            rows.append(np.concatenate(cells, 1))
        hh = max(r.shape[0] for r in rows)
        rows = [np.pad(r, ((0, hh - r.shape[0]), (0, 0), (0, 0))) for r in rows]
        out.append(np.concatenate(rows, 0))
    path = LOG_ROOT / "renders" / "icl-peg" / "videos" / cfg.out
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, out, fps=cfg.fps)
    print("wrote", path, out[0].shape)
    for j, e in enumerate(idx):
        print(f"task {j} (env {e}): {cmds2[e]}")


if __name__ == "__main__":
    main()
