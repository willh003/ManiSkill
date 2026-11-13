"""Stochasticity with IDENTICAL resets: every repeat uses the same reset seed (same robot qpos, peg pose/geometry, hole offset) and the same goal.
Verifies the initial obs and then every step's obs/action diverge only through the sim. Saves logs/renders/offline/stoch_fixed_bc-500.{npz,png}."""
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

N, K, NAME, SEED = 512, 32, "bc-500", 9000
dev = torch.device("cuda")
args = Args(sizes="500")
env = gym.make("PegInsertionSideBias-v1", num_envs=N, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
if isinstance(env.action_space, gym.spaces.Dict):
    env = FlattenActionSpaceWrapper(env)
venv = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)
A = venv.single_action_space.shape[0]
Tc = len(ctx_index(args.num_steps, args.ctx_stride, dev))
goals = checker_sample(N, "all", dev, torch.Generator(device=dev).manual_seed(777), args.split)
class _E:  # student mode calls the expert only for labels; unused here
    def get_action(self, o, deterministic=True): return torch.zeros(o.shape[0], A, device=dev)
S, O0, ACT = [], [], []
for s in (1, 2, 3):
    pol = Policy(A, Tc, args).to(dev)
    pol.load_state_dict(torch.load(LOG_ROOT / f"runs/offline-vis-ckpt-s{s}/checkpoints/{NAME}.pt", map_location=dev)); pol.eval()
    rs = [rollout(venv, _E(), pol, args, goals, SEED, "student", empty_ctx(N, Tc, A, dev)) for _ in range(K)]
    S.append(np.stack([r["succ"].cpu().numpy() for r in rs]))
    o0 = torch.stack([r["obs"][:, 0] for r in rs]); O0.append((o0 - o0[0]).abs().max().item())
    act = torch.stack([r["act"] for r in rs]); ACT.append(act)  # (K, N, T, A)
    print(f"policy seed {s}: pass@1 {S[-1].mean():.3f}; max |initial obs diff| across repeats {O0[-1]:.2e}", flush=True)
S = np.stack(S); G = goals.cpu().numpy(); p = S.mean(1)
# time of first action divergence between repeat 0 and the others
div = (ACT[0] - ACT[0][:1]).abs().amax(-1) > 1e-4  # (K, N, T)
first = np.where(div.any(-1).cpu().numpy(), div.float().argmax(-1).cpu().numpy(), -1)[1:]
np.savez(LOG_ROOT / f"renders/offline/stoch_fixed_{NAME}.npz", goals=G, succ=S, init_diff=np.array(O0))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
ax[0].hist(p.ravel(), bins=np.linspace(0, 1, K + 1), color="C0")
ax[0].set_xlabel(f"per-goal success fraction over {K} identical resets"); ax[0].set_ylabel("count (goals x 3 policy seeds)")
ax[0].set_title(f"{((p > 0) & (p < 1)).mean():.1%} of goals mixed; {(p == 1).mean():.0%} always succeed, {(p == 0).mean():.0%} always fail")
sc = ax[1].scatter(G[:, 0], G[:, 1], c=p[0], cmap="RdYlGn", vmin=0, vmax=1, s=18)
ax[1].set_aspect("equal"); ax[1].set_xlabel("hole x"); ax[1].set_ylabel("hole y"); ax[1].set_title("policy seed 1: success fraction per goal"); fig.colorbar(sc, ax=ax[1])
f = first[first >= 0]
ax[2].hist(f, bins=40, color="C1"); ax[2].set_xlabel("first step where actions differ from repeat 0 (> 1e-4)"); ax[2].set_ylabel("count (repeat x goal, seed 1)")
ax[2].set_title(f"{(first < 0).mean():.0%} of repeats never diverge in action")
out = LOG_ROOT / f"renders/offline/stoch_fixed_{NAME}.png"; fig.savefig(out, dpi=120); print(out)
