"""Stochasticity of the offline BC N=500 policies: same 512 goals (and same env index -> same peg geometry), K different reset seeds each.
Also a determinism check: the same reset seed repeated. Saves logs/renders/offline/stoch_bc-500.{npz,png}."""
import sys
import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from icl_checker import Policy, ctx_index, empty_ctx, rollout
from icl_offline import Args
from icl_tasks import checker_sample
from run_logging import LOG_ROOT

N, K, NAME = 512, 32, "bc-500"
dev = torch.device("cuda")
args = Args(sizes="500")
env = gym.make("PegInsertionSideBias-v1", num_envs=N, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
if isinstance(env.action_space, gym.spaces.Dict):
    env = FlattenActionSpaceWrapper(env)
venv = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)
A = venv.single_action_space.shape[0]
Tc = len(ctx_index(args.num_steps, args.ctx_stride, dev))
goals = checker_sample(N, "all", dev, torch.Generator(device=dev).manual_seed(777), args.split)
S, det = [], []
for s in (1, 2, 3):
    pol = Policy(A, Tc, args).to(dev)
    pol.load_state_dict(torch.load(LOG_ROOT / f"runs/offline-vis-ckpt-s{s}/checkpoints/{NAME}.pt", map_location=dev)); pol.eval()
    run = lambda seed: rollout(venv, None, pol, args, goals, seed, "student", empty_ctx(N, Tc, A, dev))["succ"] if False else None
    import icl_checker
    class _E:  # student mode still calls expert for labels; use a dummy returning zeros
        def get_action(self, o, deterministic=True): return torch.zeros(o.shape[0], A, device=dev)
    ex = _E()
    r = lambda seed: rollout(venv, ex, pol, args, goals, seed, "student", empty_ctx(N, Tc, A, dev))["succ"].cpu().numpy()
    S.append(np.stack([r(9000 + k) for k in range(K)]))  # (K, N)
    a, b = r(9000), r(9000)
    det.append((a == b).mean())
    print(f"policy seed {s}: pass@1 {S[-1].mean():.3f}; same-seed repeat agreement {det[-1]:.4f}", flush=True)
S = np.stack(S)  # (3, K, N)
G = goals.cpu().numpy()
np.savez(LOG_ROOT / f"renders/offline/stoch_{NAME}.npz", goals=G, succ=S, det=np.array(det))

p = S.mean(1)  # (3, N) per-goal success prob over resets
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
ax[0].hist(p.ravel(), bins=np.linspace(0, 1, K + 1), color="C0")
ax[0].set_xlabel(f"per-goal success fraction over {K} resets"); ax[0].set_ylabel("count (goals x 3 policy seeds)")
det_frac = ((p > 0) & (p < 1)).mean()
ax[0].set_title(f"{det_frac:.0%} of goals are sometimes-success, sometimes-fail")
# binary-ness: if goal fully determined success, p in {0,1}; compare with a binomial at same mean
ax[1].scatter(G[:, 0], G[:, 1], c=p[0], cmap="RdYlGn", vmin=0, vmax=1, s=18)
ax[1].set_aspect("equal"); ax[1].set_xlabel("hole x"); ax[1].set_ylabel("hole y"); ax[1].set_title("policy seed 1: success fraction per goal")
sc = ax[1].collections[0]; fig.colorbar(sc, ax=ax[1], label="success fraction")
ys = p[0][np.argsort(p[0])]
cnt = [((p[0] == 0).mean(), "always fail"), (((p[0] > 0) & (p[0] < 1)).mean(), "mixed"), ((p[0] == 1).mean(), "always succeed")]
ax[2].bar([c[1] for c in cnt], [c[0] for c in cnt], color=["C3", "C1", "C2"])
for i, c in enumerate(cnt): ax[2].text(i, c[0] + .01, f"{c[0]:.2f}", ha="center")
ax[2].set_ylabel("fraction of goals (policy seed 1)"); ax[2].set_title(f"same-seed repeat agreement: {np.mean(det):.3f}")
out = LOG_ROOT / f"renders/offline/stoch_{NAME}.png"; fig.savefig(out, dpi=120); print(out)
