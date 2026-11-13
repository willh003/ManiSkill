"""Failure modes of the offline BC N=500 policies (offline-vis-ckpt-s{1,2,3}): same 512 goals, fixed reset seed, 100 steps.
Per episode: ever grasped / grasped at end / peg-head pose in the hole frame at end and at its closest approach."""
import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import icl_checker
from icl_checker import Policy, ctx_index, empty_ctx, sobs
from icl_offline import Args
from icl_tasks import checker_sample
from run_logging import LOG_ROOT

N, T, NAME, SEED = 512, 100, "bc-500", 9000
dev = torch.device("cuda")
args = Args(sizes="500")
env = gym.make("PegInsertionSideBias-v1", num_envs=N, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
if isinstance(env.action_space, gym.spaces.Dict):
    env = FlattenActionSpaceWrapper(env)
venv = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)
A = venv.single_action_space.shape[0]
base = venv.base_env
Tc = len(ctx_index(args.num_steps, args.ctx_stride, dev))
goals = checker_sample(N, "all", dev, torch.Generator(device=dev).manual_seed(777), args.split)
low, high = (torch.from_numpy(b).to(dev) for b in (venv.single_action_space.low, venv.single_action_space.high))
out = {}
for s in (1, 2, 3):
    pol = Policy(A, Tc, args).to(dev)
    pol.load_state_dict(torch.load(LOG_ROOT / f"runs/offline-vis-ckpt-s{s}/checkpoints/{NAME}.pt", map_location=dev)); pol.eval()
    c = empty_ctx(N, Tc, A, dev)
    with torch.no_grad():
        memory, pad = pol.encode(c["s"], c["a"], c["y"], c["valid"])
    obs, _ = venv.reset(seed=SEED, options=icl_checker.TASK.reset_options(goals))
    rad = base.box_hole_radii.clone()
    H, G, SU, TCPD = [], [], [], []
    for t in range(T):
        with torch.no_grad():
            a = pol.decode(sobs(obs)[:, None], memory, pad)[:, 0]
        obs, _, _, _, info = venv.step(torch.max(torch.min(a, high), low))
        succ, head = base.has_peg_inserted()
        H.append(head.clone()); SU.append(succ.clone()); G.append(base.agent.is_grasping(base.peg, max_angle=20).clone())
        TCPD.append(torch.linalg.norm(base.agent.tcp.pose.p - base.peg.pose.p, dim=1))
    H, SU, G, TCPD = (torch.stack(x, 1).cpu().numpy() for x in (H, SU, G, TCPD))  # H (N,T,3)
    out[s] = dict(head=H, succ=SU, grasp=G, tcpd=TCPD, rad=rad.cpu().numpy())
    print(f"seed {s}: success(ever) {SU.any(1).mean():.3f}", flush=True)

cats = ["never grasped", "grasped, dropped (not held at end)", "held at end: peg head off-axis (y/z outside hole)", "held at end: aligned but not inserted deep enough (x)", "held, other"]
fig, ax = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
counts = np.zeros((3, len(cats))); lat_all, x_all = [], []
for i, (s, d) in enumerate(out.items()):
    ok = d["succ"].any(1); grasped_ever = d["grasp"].any(1); held_end = d["grasp"][:, -10:].all(1)
    head, rad = d["head"], d["rad"][:, None]
    lat = np.maximum(np.abs(head[..., 1]), np.abs(head[..., 2])) / rad  # >1: outside hole radius
    fe = ~ok
    c = np.zeros(len(cats), int)
    c[0] = (fe & ~grasped_ever).sum(); c[1] = (fe & grasped_ever & ~held_end).sum()
    held = fe & held_end
    c[2] = (held & (lat[:, -1] > 1)).sum(); c[3] = (held & (lat[:, -1] <= 1) & (head[:, -1, 0] < -0.015)).sum()
    c[4] = held.sum() - c[2] - c[3]
    counts[i] = c / N
    lat_all.append(lat[fe, -1]); x_all.append(head[fe, -1, 0])
    print(f"seed {s}: success {ok.mean():.3f}; failures " + ", ".join(f"{n}: {v / N:.3f}" for n, v in zip(cats, c)), flush=True)
m = counts.mean(0); sd = counts.std(0)
cols = ["C3", "C1", "C0", "C4", "gray"]
ax[0].barh(range(len(cats)), m, xerr=sd, color=cols); ax[0].set_yticks(range(len(cats))); ax[0].set_yticklabels([c.replace(": ", ":\n") for c in cats], fontsize=8)
ax[0].invert_yaxis(); ax[0].set_xlabel("fraction of all episodes (3 policy seeds, mean +- std)"); ax[0].set_title(f"failure modes (success {1 - m.sum():.2f})")
L = np.concatenate(lat_all); X = np.concatenate(x_all)
ax[1].hist(np.clip(L, 0, 40), bins=60, color="C0"); ax[1].axvline(1, c="k", ls="--"); ax[1].set_xlabel("final max(|y|,|z|) of peg head in hole frame / hole radius (clipped at 40)"); ax[1].set_title("failed episodes: lateral error of peg head at end")
ax[2].scatter(X, np.clip(L, 0, 40), s=4, alpha=.3); ax[2].axvline(-0.015, c="k", ls="--"); ax[2].axhline(1, c="k", ls="--")
ax[2].set_xlabel("final peg-head x in hole frame (m; >= -0.015 counts as inserted)"); ax[2].set_ylabel("lateral error / radius (clipped)"); ax[2].set_title("failed episodes: where the head ends up")
fig.savefig(LOG_ROOT / f"renders/offline/failmodes_{NAME}.png", dpi=120)
np.savez(LOG_ROOT / f"renders/offline/failmodes_{NAME}.npz", counts=counts, lat=L, x=X)
print("fail x quantiles", np.percentile(X, [5, 25, 50, 75, 95]).round(3), "lat quantiles", np.percentile(L, [5, 25, 50, 75, 95]).round(1))
