"""What does the feedback look like, and does its distribution shift across rounds / attempts? For one icl_onpolicy.py run.
Feedback of a rollout = c = g - f (g = the task's expert final peg pose, f = the rollout's final peg pose); shown as the
position part in cm and the orientation error in degrees.
 A. training pool on the demo tasks (state.pt): round 0 = visible-hole BC rollouts, round r = ICL round-r rollouts (all K attempts)
 B. eval tasks, attempt 1 = BC then round-r ICL (as icl_onpolicy_bcfirst.py): feedback after each attempt, rounds 1, 2, 8
 C. how many test-time context inputs hit the +-5 std clip of each round's normaliser (TF.forward clamps there)
Usage: python icl_feedback_viz.py RUN_DIR  -> logs/renders/icl-feedback/<run>/*.png + feedback.json"""
import json
import os
import sys

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from icl_checker import rollout
from icl_tasks import checker_sample
from icl_transformer import LAYOUT, PEG, TF, final_idx, hist_of, mk, quat_angle
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo import Agent
from run_logging import LOG_ROOT

INK, MUTE, BG = "#0b0b0b", "#52514e", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("b", ["#b9d6f5", "#2a78d6", "#0c2f66"])
BCC, FAIL, OK = "#eb6834", "#2a78d6", "#b5b3ad"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTE, "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE, "figure.facecolor": BG, "axes.facecolor": BG})


def feats(f, g):
    """-> dict of numpy: c position (cm), |c| (cm), orientation error (deg)."""
    c = (g[:, :3] - f[:, :3]) * 100
    return dict(c=c.cpu().numpy(), n=c.norm(dim=-1).cpu().numpy(), ang=quat_angle(f[:, 3:], g[:, 3:]).cpu().numpy())


def style(a, title=None):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(alpha=.22, lw=.6)
    if title:
        a.set_title(title, fontsize=9.5, loc="left", color=INK)


def head(f, title, sub):
    f.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    f.text(0.01, 0.935, sub, fontsize=8.5, color=MUTE)


def cdf(a, x, col, lab, ls="-", lw=2):
    x = np.sort(np.clip(x, 0.1, None))
    a.plot(x, np.arange(1, len(x) + 1) / len(x), ls, color=col, lw=lw, label=lab)


def main(rd):
    dev = torch.device("cuda")
    name = rd.rstrip("/").split("/")[-1]
    out = f"{LOG_ROOT}/renders/icl-feedback/{name}"
    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(f"{rd}/config.json"))["args"]
    S = torch.load(f"{rd}/state.pt", map_location=dev, weights_only=False)
    kind, L, R, K, N = cfg["kind"], LAYOUT[cfg["kind"]], cfg["rounds"], cfg["K"], cfg["num_envs"]
    EO, EA = S["EO"], S["EA"]
    e_peg = EO[torch.arange(len(EO)), final_idx(EA)][:, PEG]
    pool = S["pool"]
    rec = {}

    # ---- A. training pool on the demo tasks, by the round that produced it
    srcs = sorted(set(pool["rnd"].long().tolist()))
    A_ = {}
    for r in srcs:
        m = pool["rnd"].long() == r
        A_[r] = dict(feats(pool["peg"][m], e_peg[pool["task"][m]]), succ=pool["succ"][m].cpu().numpy(), att=pool["att"][m].long().cpu().numpy())
    rec["pool"] = {str(r): dict(n=len(v["n"]), succ=float(v["succ"].mean()), median_cm=float(np.median(v["n"])), median_cm_fail=float(np.median(v["n"][~v["succ"]])) if (~v["succ"]).any() else None,
                                median_deg_fail=float(np.median(v["ang"][~v["succ"]])) if (~v["succ"]).any() else None) for r, v in A_.items()}
    lab = lambda r: "round 0: visible-hole BC" if r == 0 else f"round {r}: ICL, {K} attempts"
    lim = 60
    fig, axs = plt.subplots(2, 4, figsize=(13, 6.6), sharex=True, sharey=True)
    for a, r in zip(axs.flat, srcs):
        v = A_[r]
        s = v["succ"].astype(bool)
        a.scatter(np.clip(v["c"][s, 0], -lim, lim), np.clip(v["c"][s, 1], -lim, lim), s=5, color=OK, alpha=.5, lw=0, label="success")
        a.scatter(np.clip(v["c"][~s, 0], -lim, lim), np.clip(v["c"][~s, 1], -lim, lim), s=5, color=BCC if r == 0 else FAIL, alpha=.45, lw=0, label="failure")
        a.set_xlim(-lim, lim), a.set_ylim(-lim, lim), a.set_aspect("equal")
        style(a, f"{lab(r)}\nn={len(s)}, success {s.mean():.2f}")
    for a in axs[-1]:
        a.set_xlabel("c_x (cm)")
    for a in axs[:, 0]:
        a.set_ylabel("c_y (cm)")
    axs.flat[0].legend(fontsize=8, frameon=False, loc="lower left", markerscale=3)
    head(fig, f"Feedback on the demo tasks, by the round that produced it ({cfg['demos']} demos)",
         f"Training pool of {name}, top view: c = expert final peg position - rollout final peg position. Clipped to +-{lim} cm. Round r's model trains on rounds 0..r-1.")
    fig.tight_layout(rect=(0, 0, 1, 0.92)), fig.savefig(f"{out}/A_pool_scatter.png", dpi=130), plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for r in srcs:
        v, col = A_[r], (BCC if r == 0 else SEQ((r - 1) / max(len(srcs) - 2, 1)))
        fl = ~v["succ"].astype(bool)
        cdf(axs[0], v["n"][fl], col, lab(r)), cdf(axs[1], v["ang"][fl], col, lab(r))
    axs[0].set_xscale("log"), axs[0].set_xlabel("|c| position error of failed rollouts (cm, log)"), axs[0].set_ylabel("fraction of failed rollouts")
    axs[1].set_xlabel("orientation error of failed rollouts (deg)")
    for a in axs:
        style(a)
    axs[1].legend(fontsize=8, frameon=False, loc="lower right")
    head(fig, f"Feedback size on the demo tasks, failures only ({cfg['demos']} demos)", f"{name}. Cumulative distribution per round. Successes (c ~ 0) left out.")
    fig.tight_layout(rect=(0, 0, 1, 0.9)), fig.savefig(f"{out}/A_pool_cdf.png", dpi=130), plt.close(fig)

    # ---- B/C. eval tasks: BC attempt 1, then round-r ICL with a sliding window
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
    g = fin(rollout(venv, expert, None, args, goals, cfg["eval_seed"], "expert"))
    ctx_of = lambda y, v: dict(s=torch.zeros(N, 1, 43, device=dev), a=torch.zeros(N, 1, Ad, device=dev), y=y.reshape(N, 1, -1), valid=torch.full((N, 1), v, dtype=torch.bool, device=dev))
    bc = TF(Ad, 0, cfg["d"], cfg["layers"], cfg["heads"]).to(dev)
    bc.load_state_dict(torch.load(f"{rd}/checkpoints/tf-bc.pt", map_location=dev)), bc.eval()
    r1 = rollout(venv, expert, bc, args, goals, cfg["eval_seed"], "student", ctx_of(torch.zeros(N, 7, device=dev), False), "normal", mask_pose=False)
    B1 = dict(feats(fin(r1), g), succ=r1["succ"].cpu().numpy())
    ctx1 = mk(kind, hist_of(r1["obs"], final_idx(r1["alive"])), fin(r1), g, ones)
    rec["eval_bc"] = dict(succ=float(B1["succ"].mean()), median_cm_fail=float(np.median(B1["n"][~B1["succ"]])))
    chains, clip = {}, {}
    for rnd in range(1, R + 1):
        pol = TF(Ad, L, cfg["d"], cfg["layers"], cfg["heads"], 0.0, True).to(dev)
        pol.load_state_dict(torch.load(f"{rd}/checkpoints/round{rnd}.pt", map_location=dev)), pol.eval()
        z = lambda c: ((c - pol.c_mean) / pol.c_std).abs().reshape(N, -1)
        r, ch, zs = r1, [], [z(ctx1)]
        for k in range(1, K):
            c = mk(kind, hist_of(r["obs"], final_idx(r["alive"])), fin(r), g, ones)
            if k > 1:
                zs.append(z(c))
            r = rollout(venv, expert, pol, args, goals, cfg["eval_seed"], "student", ctx_of(c, True), "normal", mask_pose=True)
            ch.append(dict(feats(fin(r), g), succ=r["succ"].cpu().numpy()))
        chains[rnd] = ch
        zs = torch.stack(zs)  # (K-1 contexts, N, dims)
        # also: the round's own training contexts (pool rounds 0..rnd-1)
        m = pool["rnd"].long() < rnd
        zt = z(mk(kind, pool["hist"][m], pool["peg"][m], e_peg[pool["task"][m]], torch.ones(int(m.sum()), device=dev)))
        clip[rnd] = dict(bc_ctx=float((zs[0] > 5).any(-1).float().mean()), icl_ctx=float((zs[1:] > 5).any(-1).float().mean()),
                         train_ctx=float((zt > 5).any(-1).float().mean()), max_z_bc=float(zs[0].max()), c_std_pos_cm=(pol.c_std[-1, :3] * 100).tolist())
        print(f"round {rnd}: clip bc {clip[rnd]['bc_ctx']:.3f} icl {clip[rnd]['icl_ctx']:.3f} train {clip[rnd]['train_ctx']:.3f}  c_std(pos, cm) {[round(x, 1) for x in clip[rnd]['c_std_pos_cm']]}", flush=True)
    rec["clip"] = clip
    rec["eval_chain"] = {str(rnd): [dict(attempt=k + 2, succ=float(v["succ"].mean()), median_cm_fail=float(np.median(v["n"][~v["succ"]])) if (~v["succ"]).any() else None) for k, v in enumerate(ch)] for rnd, ch in chains.items()}

    show = [x for x in (1, 2, R) if x in chains]
    fig, axs = plt.subplots(1, len(show), figsize=(5 * len(show), 4.6), sharey=True)
    for a, rnd in zip(np.atleast_1d(axs), show):
        cdf(a, B1["n"][~B1["succ"]], BCC, f"after attempt 1 (BC), succ {B1['succ'].mean():.2f}", lw=2.4)
        for k, v in enumerate(chains[rnd][:-1]):
            cdf(a, v["n"][~v["succ"]], SEQ(k / max(K - 3, 1)), f"after attempt {k + 2} (ICL), succ {v['succ'].mean():.2f}", lw=1.6)
        a.set_xscale("log"), a.set_xlabel("|c| of failed attempts (cm, log)")
        style(a, f"round {rnd} ICL model")
        a.legend(fontsize=7.5, frameon=False, loc="lower right")
    np.atleast_1d(axs)[0].set_ylabel("fraction of failed attempts")
    head(fig, f"Feedback after each attempt on the 512 eval tasks ({cfg['demos']} demos)", f"{name}. Attempt 1 = visible-hole BC, then the round's ICL model with a sliding window. This is the feedback the next attempt sees.")
    fig.tight_layout(rect=(0, 0, 1, 0.88)), fig.savefig(f"{out}/B_eval_attempts_cdf.png", dpi=130), plt.close(fig)

    fig, axs = plt.subplots(1, len(show) + 1, figsize=(4.2 * (len(show) + 1), 4.4), sharex=True, sharey=True)
    pan = [("visible-hole BC, attempt 1", B1, BCC)] + [(f"round {rnd} ICL, attempt 2", chains[rnd][0], FAIL) for rnd in show]
    for a, (t, v, col) in zip(axs, pan):
        s = v["succ"].astype(bool)
        a.scatter(np.clip(v["c"][s, 0], -lim, lim), np.clip(v["c"][s, 1], -lim, lim), s=5, color=OK, alpha=.5, lw=0)
        a.scatter(np.clip(v["c"][~s, 0], -lim, lim), np.clip(v["c"][~s, 1], -lim, lim), s=5, color=col, alpha=.5, lw=0)
        a.set_xlim(-lim, lim), a.set_ylim(-lim, lim), a.set_aspect("equal"), a.set_xlabel("c_x (cm)")
        style(a, f"{t}\nsuccess {s.mean():.2f}")
    axs[0].set_ylabel("c_y (cm)")
    head(fig, f"Feedback on the eval tasks, top view ({cfg['demos']} demos)", f"{name}. Grey = success, colour = failure. Clipped to +-{lim} cm.")
    fig.tight_layout(rect=(0, 0, 1, 0.86)), fig.savefig(f"{out}/B_eval_scatter.png", dpi=130), plt.close(fig)

    fig, a = plt.subplots(figsize=(7.2, 4.4))
    x = np.array(sorted(clip))
    for key, col, lb in (("train_ctx", OK, "training contexts (pool rounds 0..r-1)"), ("bc_ctx", BCC, "test: BC attempt-1 feedback"), ("icl_ctx", FAIL, "test: ICL attempt feedback")):
        a.plot(x, [clip[r][key] for r in x], "-o", color=col, lw=2, ms=4.5, label=lb)
    a.set_xlabel("round"), a.set_ylabel("fraction of contexts with an input clipped"), a.set_ylim(0, 1)
    style(a), a.legend(fontsize=8.5, frameon=False)
    head(fig, "Context inputs beyond the +-5 std clip", f"{name}. Normaliser = mean/std of the round's training contexts; any of the {L * 7} inputs past 5 std counts.")
    fig.tight_layout(rect=(0, 0, 1, 0.88)), fig.savefig(f"{out}/C_clip.png", dpi=130), plt.close(fig)
    json.dump(rec, open(f"{out}/feedback.json", "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
