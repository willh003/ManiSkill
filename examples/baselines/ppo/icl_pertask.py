"""Which eval tasks succeed on which attempt? Per-task success over K attempts for visible-hole BC (no noise, noise 0.05) and an
icl_onpolicy.py round (attempt 1 = the run's BC, then ICL with a sliding window). Same 512 eval tasks as the run.
Usage: python icl_pertask.py RUN_DIR ROUND  -> logs/renders/icl-pertask/<run>-r<ROUND>/{pertask.pt, *.png}"""
import json
import os
import sys

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from icl_checker import POSE, rollout
from icl_tasks import checker_sample
from icl_transformer import LAYOUT, PEG, TF, final_idx, hist_of, mk
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo import Agent
from run_logging import LOG_ROOT

INK, MUTE, BG, OKC, FAILC = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#e4e2dc"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTE, "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE, "figure.facecolor": BG, "axes.facecolor": BG})


def main(rd, rnd):
    dev = torch.device("cuda")
    name = rd.rstrip("/").split("/")[-1]
    out = f"{LOG_ROOT}/renders/icl-pertask/{name}-r{rnd}"
    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(f"{rd}/config.json"))["args"]
    kind, L, K, N = cfg["kind"], LAYOUT[cfg["kind"]], cfg["K"], cfg["num_envs"]

    class A0:
        pass

    args = A0()
    args.num_envs, args.num_steps = N, cfg["num_steps"]
    env = gym.make("PegInsertionSideBias-v1", num_envs=N, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
    env = FlattenActionSpaceWrapper(env) if isinstance(env.action_space, gym.spaces.Dict) else env
    venv = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)
    Ad = venv.single_action_space.shape[0]

    class _S:
        single_observation_space = type("S", (), {"shape": (43,)})
        single_action_space = venv.single_action_space

    expert = Agent(_S).to(dev)
    expert.load_state_dict(torch.load(f"{LOG_ROOT}/runs/" + cfg["expert_checkpoint"].split("/runs/")[-1], map_location=dev))
    expert.eval()
    ar, ones = torch.arange(N, device=dev), torch.ones(N, device=dev)
    fin = lambda r: r["obs"][ar, final_idx(r["alive"])][:, PEG]
    goals = checker_sample(N, "all", dev, torch.Generator(device=dev).manual_seed(cfg["eval_seed"]), cfg["split"])
    ex = rollout(venv, expert, None, args, goals, cfg["eval_seed"], "expert")
    g = ex["obs"][:, 0, POSE] if cfg.get("goal_ref", "peg") == "hole" else fin(ex)
    hole_xy = ex["obs"][:, 0, POSE][:, :2].cpu().numpy() * 100
    ctx_of = lambda y, v: dict(s=torch.zeros(N, 1, 43, device=dev), a=torch.zeros(N, 1, Ad, device=dev), y=y.reshape(N, 1, -1), valid=torch.full((N, 1), v, dtype=torch.bool, device=dev))
    bc = TF(Ad, 0, cfg["d"], cfg["layers"], cfg["heads"]).to(dev)
    bc.load_state_dict(torch.load(f"{rd}/checkpoints/tf-bc.pt", map_location=dev)), bc.eval()
    z7 = ctx_of(torch.zeros(N, 7, device=dev), False)
    S = {}
    for tag, noise in (("BC, no noise", 0.0), ("BC + noise 0.05", 0.05)):
        S[tag] = torch.stack([rollout(venv, expert, bc, args, goals, cfg["eval_seed"], "student", z7, "normal", mask_pose=False, act_noise=noise)["succ"] for _ in range(K)], 1)
    pol = TF(Ad, L, cfg["d"], cfg["layers"], cfg["heads"], 0.0, True).to(dev)
    pol.load_state_dict(torch.load(f"{rd}/checkpoints/round{rnd}.pt", map_location=dev)), pol.eval()
    r = rollout(venv, expert, bc, args, goals, cfg["eval_seed"], "student", z7, "normal", mask_pose=False)
    su = [r["succ"]]
    for k in range(1, K):
        c = mk(kind, hist_of(r["obs"], final_idx(r["alive"])), fin(r), g, ones)
        r = rollout(venv, expert, pol, args, goals, cfg["eval_seed"], "student", ctx_of(c, True), "normal", mask_pose=True)
        su.append(r["succ"])
    S[f"ours r{rnd}: attempt 1 = BC, then ICL"] = torch.stack(su, 1)
    S = {k: v.cpu().numpy().astype(bool) for k, v in S.items()}
    torch.save(dict(succ=S, hole_xy=hole_xy, expert=ex["succ"].cpu().numpy()), f"{out}/pertask.pt")
    stats = {}
    for k, v in S.items():
        stats[k] = dict(perf=v.mean(0).round(3).tolist(), pass_k=np.maximum.accumulate(v, 1).mean(0).round(3).tolist(),
                        always=float(v.all(1).mean()), never=float((~v).all(1).mean()), sometimes=float((v.any(1) & ~v.all(1)).mean()),
                        same_as_prev=float((v[:, 1:] == v[:, :-1]).mean()))
        print(f"{k:40s} perf {v.mean():.3f} pass@{K} {v.any(1).mean():.3f} | always {stats[k]['always']:.3f} sometimes {stats[k]['sometimes']:.3f} never {stats[k]['never']:.3f} | attempt k same outcome as k-1: {stats[k]['same_as_prev']:.3f}", flush=True)
    json.dump(stats, open(f"{out}/stats.json", "w"), indent=1)

    # 1. per-task outcome matrix: tasks sorted by number of successes, then by first success
    names = list(S)
    f, axs = plt.subplots(1, len(names), figsize=(4.2 * len(names), 6.2), sharey=True)
    for a, k in zip(axs, names):
        v = S[k]
        order = np.lexsort((np.argmax(v, 1), -v.sum(1)))
        a.imshow(v[order], aspect="auto", interpolation="nearest", cmap=matplotlib.colors.ListedColormap([FAILC, OKC]), extent=(0.5, K + 0.5, len(v), 0))
        a.set_xticks(range(1, K + 1)), a.set_xlabel("attempt k")
        st = stats[k]
        a.set_title(f"{k}\nalways {st['always']:.2f} | sometimes {st['sometimes']:.2f} | never {st['never']:.2f}", loc="left", fontsize=9.5, color=INK)
    axs[0].set_ylabel("eval task (sorted by number of successes)")
    f.suptitle("Which tasks succeed on which attempt", x=0.01, y=0.995, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    f.text(0.01, 0.955, f"{name}, round {rnd}, 512 eval tasks, one row per task. Blue = success on that attempt. 'sometimes' tasks are what make pass@k exceed perf@k.", fontsize=8.5, color=MUTE, va="top")
    f.tight_layout(rect=(0, 0, 1, 0.93)), f.savefig(f"{out}/1_task_by_attempt.png", dpi=130), plt.close(f)

    # 2. goals (hole x, y) per attempt
    f, axs = plt.subplots(len(names), K, figsize=(2.05 * K, 2.25 * len(names) + 0.9), sharex=True, sharey=True)
    for i, k in enumerate(names):
        for j in range(K):
            a, s = axs[i, j], S[k][:, j]
            a.scatter(hole_xy[~s, 0], hole_xy[~s, 1], s=4, color="#c9c6be", lw=0)
            a.scatter(hole_xy[s, 0], hole_xy[s, 1], s=4, color=OKC, lw=0)
            a.set_aspect("equal"), a.tick_params(labelsize=7)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            if i == 0:
                a.set_title(f"attempt {j + 1}", fontsize=9, color=INK)
            a.text(0.02, 0.02, f"{s.mean():.2f}", transform=a.transAxes, fontsize=7.5, color=INK)
        axs[i, 0].set_ylabel(k.replace(": ", ":\n"), fontsize=8.5)
    f.suptitle("Goals that succeed (blue) or fail (grey) on each attempt", x=0.01, y=0.995, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    f.text(0.01, 1 - 0.5 / f.get_figheight(), f"{name}, round {rnd}. Each dot is an eval task at its hole position (x, y in cm); number = success rate of that attempt.", fontsize=8.5, color=MUTE, va="top")
    f.tight_layout(rect=(0, 0, 1, 1 - 0.75 / f.get_figheight())), f.savefig(f"{out}/2_goals_by_attempt.png", dpi=130), plt.close(f)

    # 3. goals coloured by number of successes out of K
    f, axs = plt.subplots(1, len(names), figsize=(4.3 * len(names), 4.6), sharex=True, sharey=True)
    for a, k in zip(axs, names):
        n = S[k].sum(1)
        sc = a.scatter(hole_xy[:, 0], hole_xy[:, 1], c=n, s=9, cmap="Blues", vmin=-1, vmax=K, lw=0.2, edgecolor="#9a978f")
        a.set_aspect("equal"), a.set_title(k, loc="left", fontsize=9.5, color=INK), a.set_xlabel("hole x (cm)")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    axs[0].set_ylabel("hole y (cm)")
    cb = f.colorbar(sc, ax=list(axs), shrink=0.85, pad=0.01)
    cb.set_label(f"successes out of {K}")
    f.suptitle(f"How many of the {K} attempts succeed, per goal", x=0.01, y=0.99, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    f.savefig(f"{out}/3_goals_success_count.png", dpi=130, bbox_inches="tight"), plt.close(f)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
