"""Failure modes + stochasticity of the PickCube BC policies (logs/runs/pick-bc/checkpoints/bc-{N}-s{seed}.pt), 3 policy seeds.
Failure modes: 512 fixed resets (eval seed), per-step grasp / cube-goal distance / tcp-cube distance / static from the env.
Stochasticity: the SAME reset (identical initial state) repeated K times (PickCube resets randomise cube + goal, so there is no 'same pose, new reset' to repeat).
Usage: python pick_bc_analysis.py 50 100"""
import sys
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from run_logging import LOG_ROOT

N, T, K, EVAL_SEED = 512, 50, 32, 12345
dev = torch.device("cuda")
env = gym.make("PickCube-v1", num_envs=N, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
if isinstance(env.action_space, gym.spaces.Dict):
    env = FlattenActionSpaceWrapper(env)
venv = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)
base = venv.base_env
S, A = venv.single_observation_space.shape[0], venv.single_action_space.shape[0]
low, high = (torch.from_numpy(b).to(dev) for b in (venv.single_action_space.low, venv.single_action_space.high))
thresh = base.goal_thresh


def load(n, s):
    ck = torch.load(LOG_ROOT / f"runs/pick-bc/checkpoints/bc-{n}-s{s}.pt", map_location=dev)
    net = nn.Sequential(nn.Linear(S, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, A)).to(dev)
    net.load_state_dict(ck["net"])
    return lambda o: net((o - ck["mu"]) / ck["sd"])


def run(pol, seed):
    obs, _ = venv.reset(seed=seed)
    init = dict(cube=base.cube.pose.p[:, :2].clone(), goal=base.goal_site.pose.p.clone())
    rec = {k: [] for k in ("grasp", "dgoal", "dtcp", "succ", "placed", "act")}
    for t in range(T):
        with torch.no_grad():
            a = torch.max(torch.min(pol(obs), high), low)
        obs, _, _, _, info = venv.step(a)
        i = info["final_info"] if "final_info" in info else info
        rec["grasp"].append(base.agent.is_grasping(base.cube)); rec["succ"].append(i["success"].bool()); rec["placed"].append(i["is_obj_placed"].bool())
        rec["dgoal"].append(torch.linalg.norm(base.goal_site.pose.p - base.cube.pose.p, dim=1)); rec["dtcp"].append(torch.linalg.norm(base.agent.tcp.pose.p - base.cube.pose.p, dim=1)); rec["act"].append(a)
    out = {k: torch.stack(v, 1).cpu().numpy() for k, v in rec.items()}
    out.update({k: v.cpu().numpy() for k, v in init.items()})
    return out


cats = ["never reached cube (tcp stays > 5 cm)", "reached, never grasped", "grasped then lost (not held at end)", "held at end, cube not at goal", "cube reached goal, robot not static"]
for n in [int(x) for x in sys.argv[1:]] or [50, 100]:
    counts = np.zeros((3, len(cats))); fd, ci, gi, okl, dmin = [], [], [], [], []
    stoch = []
    for si, s in enumerate((1, 2, 3)):
        pol = load(n, s)
        d = run(pol, EVAL_SEED)
        ok = d["succ"].any(1); fe = ~ok
        reached = (d["dtcp"].min(1) < 0.05); grasped = d["grasp"].any(1); held = d["grasp"][:, -5:].all(1); placed = d["placed"].any(1)
        c = np.zeros(len(cats), int)
        c[4] = (fe & placed).sum()
        rest = fe & ~placed
        c[0] = (rest & ~reached).sum(); c[1] = (rest & reached & ~grasped).sum(); c[2] = (rest & grasped & ~held).sum(); c[3] = (rest & held).sum()
        counts[si] = c / N
        fd.append(d["dgoal"][fe, -1]); ci.append(d["cube"]); okl.append(ok); gi.append(np.linalg.norm(d["goal"][:, :2] - d["cube"], axis=1))
        print(f"N={n} seed {s}: success {ok.mean():.3f}; " + ", ".join(f"{k.split(' (')[0]}: {v / N:.3f}" for k, v in zip(cats, c)), flush=True)
        # stochasticity: same reset K times
        rs = [run(pol, EVAL_SEED) for _ in range(K)]
        sc = np.stack([r["succ"].any(1) for r in rs])  # (K, N)
        a0 = rs[0]["act"]; div = [(np.abs(r["act"] - a0).max(-1) > 1e-4) for r in rs[1:]]
        first = np.stack([np.where(x.any(1), x.argmax(1), -1) for x in div])
        stoch.append((sc.mean(0), first))
        print(f"   same-reset x{K}: always succeed {(sc.mean(0) == 1).mean():.3f}, always fail {(sc.mean(0) == 0).mean():.3f}, mixed {((sc.mean(0) > 0) & (sc.mean(0) < 1)).mean():.3f}; never-diverge repeats {(first < 0).mean():.2f}", flush=True)
    fig, ax = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    m, sdv = counts.mean(0), counts.std(0)
    ax[0, 0].barh(range(len(cats)), m, xerr=sdv, color=["C3", "C1", "C0", "C4", "gray"]); ax[0, 0].set_yticks(range(len(cats))); ax[0, 0].set_yticklabels(cats, fontsize=8); ax[0, 0].invert_yaxis()
    ax[0, 0].set_xlabel("fraction of all episodes (mean +- std over 3 policy seeds)"); ax[0, 0].set_title(f"N={n}: failure modes (success {1 - m.sum():.2f})")
    ax[0, 1].hist(np.clip(np.concatenate(fd), 0, 0.5), bins=50); ax[0, 1].axvline(thresh, c="k", ls="--"); ax[0, 1].set_xlabel("final cube-goal distance (m, clipped at 0.5), failures"); ax[0, 1].set_title("how far the cube ends from the goal")
    C = np.concatenate(ci); O = np.concatenate(okl).astype(float); Gd = np.concatenate(gi)
    H, xe, ye = np.histogram2d(C[:, 0], C[:, 1], (6, 6), weights=O); Hc, _, _ = np.histogram2d(C[:, 0], C[:, 1], (6, 6))
    im = ax[0, 2].imshow((H / np.maximum(Hc, 1)).T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="RdYlGn", vmin=0, vmax=1); fig.colorbar(im, ax=ax[0, 2], label="success rate")
    ax[0, 2].set_xlabel("initial cube x"); ax[0, 2].set_ylabel("initial cube y"); ax[0, 2].set_title("success vs initial cube xy (3 seeds pooled)")
    bins = np.quantile(Gd, np.linspace(0, 1, 7)); idx = np.clip(np.digitize(Gd, bins) - 1, 0, 5)
    ax[1, 0].plot([Gd[idx == b].mean() for b in range(6)], [O[idx == b].mean() for b in range(6)], "o-"); ax[1, 0].set_ylim(0, 1); ax[1, 0].set_xlabel("initial cube-goal xy distance (m)"); ax[1, 0].set_ylabel("success rate"); ax[1, 0].set_title("success vs distance the cube must travel")
    P = np.concatenate([x[0] for x in stoch])
    ax[1, 1].hist(P, bins=np.linspace(0, 1, K + 1)); ax[1, 1].set_xlabel(f"per-reset success fraction over {K} identical repeats"); ax[1, 1].set_ylabel("count (resets x 3 seeds)")
    ax[1, 1].set_title(f"same reset repeated: {((P > 0) & (P < 1)).mean():.1%} mixed, {(P == 1).mean():.0%} always succeed, {(P == 0).mean():.0%} always fail")
    F = np.concatenate([x[1].ravel() for x in stoch]); ax[1, 2].hist(F[F >= 0], bins=40, color="C1"); ax[1, 2].set_xlabel("first step where actions differ from repeat 0 (> 1e-4)"); ax[1, 2].set_title(f"{(F < 0).mean():.0%} of repeats never diverge in action")
    fig.savefig(LOG_ROOT / f"renders/pick-bc/analysis_N{n}.png", dpi=110); print("saved", n, flush=True)
