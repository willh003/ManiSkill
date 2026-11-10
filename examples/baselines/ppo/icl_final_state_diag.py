"""Diagnose how pose-context ICL students use [final peg f, correction c = e - f]. Evaluates attempt 2 with different f given the same e.
Usage: python icl_final_state_diag.py run_dir N1,N2,..  (run_dir from icl_final_state.py --ctx-kind pose; uses checkpoints/icl-N*.pt and rollouts.pt)"""
import sys, json
import gymnasium as gym, torch
import icl_checker as ic
from icl_checker import apply_mask, rollout
from icl_tasks import checker_sample
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ppo import Agent
from run_logging import LOG_ROOT
from icl_final_state import MLP, Args, PEG, final_idx

run = sys.argv[1]; ns = [int(x) for x in sys.argv[2].split(",")]
args = Args(**{k: v for k, v in json.load(open(f"{run}/config.json"))["args"].items()})
dev = torch.device("cuda"); N = args.num_envs
env = gym.make("PegInsertionSideBias-v1", num_envs=N, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
env = FlattenActionSpaceWrapper(env) if isinstance(env.action_space, gym.spaces.Dict) else env
venv = ManiSkillVectorEnv(env, N, ignore_terminations=True, record_metrics=True)
A = venv.single_action_space.shape[0]
class _S:
    single_observation_space = type("S", (), {"shape": (43,)}); single_action_space = venv.single_action_space
expert = Agent(_S).to(dev); expert.load_state_dict(torch.load(args.expert_checkpoint, map_location=dev)); expert.eval()
full = lambda n, gen=None: checker_sample(n, "all", dev, gen, args.split)
dummy = lambda n, d: dict(s=torch.zeros(n, 1, 43, device=dev), a=torch.zeros(n, 1, A, device=dev), y=torch.zeros(n, 1, d, device=dev), valid=torch.zeros(n, 1, dtype=torch.bool, device=dev))
sc = lambda e: e["succ"].float().mean().item()
g = torch.Generator(device=dev).manual_seed(args.eval_seed); goals = full(N, g)
ex = rollout(venv, expert, None, args, goals, args.eval_seed, "expert")
ep = ex["obs"][torch.arange(N), final_idx(ex["alive"])][:, PEG]
R = torch.load(f"{run}/rollouts.pt"); Fp = R["F_peg"].to(dev)
gb = torch.Generator(device=dev).manual_seed(0)
out = {}
for n in ns:
    pol = MLP(A, 14, args.hidden).to(dev); pol.load_state_dict(torch.load(f"{run}/checkpoints/icl-N{n}.pt", map_location=dev)); pol.eval()
    e1 = rollout(venv, expert, pol, args, goals, args.eval_seed, "student", dummy(N, 14), mask_pose=True)
    f1 = e1["obs"][torch.arange(N), final_idx(e1["alive"])][:, PEG]
    train_f = Fp[:n]
    variants = {
        "A attempt1 f (standard)": f1,
        "B random train f (in-dist, same N pool)": train_f[torch.randint(0, n, (N,), device=dev, generator=gb)],
        "C random f from all 30k BC rollouts": Fp[torch.randint(0, len(Fp), (N,), device=dev, generator=gb)],
        "D constant f = mean train f": train_f.mean(0, keepdim=True).expand(N, 7),
    }
    r = {}
    for k, f in variants.items():
        c = torch.cat([f, ep - f], -1)
        ctx = dict(s=torch.zeros(N, 1, 43, device=dev), a=torch.zeros(N, 1, A, device=dev), y=c[:, None], valid=torch.ones(N, 1, dtype=torch.bool, device=dev))
        r[k] = sc(rollout(venv, expert, pol, args, goals, args.eval_seed, "student", ctx, "normal", mask_pose=True))
    r["f1 mean xy"] = f1[:, :2].mean(0).tolist(); r["f1 std xy"] = f1[:, :2].std(0).tolist()
    r["train f mean xy"] = train_f[:, :2].mean(0).tolist(); r["train f std xy"] = train_f[:, :2].std(0).tolist()
    out[n] = r
    print(n, json.dumps(r), flush=True)
json.dump(out, open(f"{run}/diag.json", "w"), indent=1)
