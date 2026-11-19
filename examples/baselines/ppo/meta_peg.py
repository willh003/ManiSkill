"""Multi-episode hidden-hole peg env for recurrent RL (plan: docs/claude/ideas/2026-11-17-rl2-peg-upper-bound-plan.md).

A meta-episode = n_ep episodes of 100 steps on the same task (box pose), all envs in lockstep. The hole pose is hidden from the
actor. After each episode the feedback c = hole pose - f (7-dim, raw pose difference as in icl_transformer.mk) is computed, where
f = peg pose at the first success of the episode (or at its last step); optional Gaussian noise on c (xy in m, yaw in rad applied
to the quaternion part through a z-rotation of the true hole quat). The actor obs is
   [student obs (clean obs 0:35, no hole pose / radius), ep one-hot (n_ep), t/100, f (7), c (7), fb_valid]
and holds the feedback of the last finished episode until the next one finishes. The critic obs is the same plus the true hole (43-dim clean obs).
same_init: every episode of a meta-episode is reset with the same seed (same peg / robot start), as in the ICL evals."""
import gymnasium as gym
import torch

from icl_tasks import checker_sample
from mani_skill.envs.tasks.tabletop.peg_insertion_side_bias import observed_hole_obs  # noqa: F401  (registers the env)
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

PEG, HOLE, EP_LEN = slice(25, 32), slice(35, 42), 100
S_DIM = 35


def zrot(yaw):
    h = yaw / 2
    z = torch.zeros_like(h)
    return torch.stack([h.cos(), z, z, h.sin()], -1)


class MetaPeg:
    def __init__(self, num_envs, n_ep=2, fb_noise_xy=0.0, fb_noise_yaw=0.0, split="wide_corner", same_init=True, device="cuda", hide_f=False, full_obs=False, train_region="all"):
        env = gym.make("PegInsertionSideBias-v1", num_envs=num_envs, obs_mode="state", control_mode="pd_joint_delta_pos",
                       sim_backend="physx_cuda", reward_mode="normalized_dense", max_episode_steps=10 ** 6)
        env = FlattenActionSpaceWrapper(env) if isinstance(env.action_space, gym.spaces.Dict) else env
        self.venv = ManiSkillVectorEnv(env, num_envs, auto_reset=False, ignore_terminations=True, record_metrics=False)
        self.N, self.n_ep, self.split, self.same_init = num_envs, n_ep, split, same_init
        self.train_region = train_region  # region of the split that training tasks are sampled from: "all" | "white" (ID); "black" is held out as OOD
        self.nxy, self.nyaw = fb_noise_xy, fb_noise_yaw
        self.hide_f = hide_f  # True: the actor obs does not contain the final peg pose f (only c = hole - f + noise); f must be remembered from the last frames
        self.dev = torch.device(device)
        self.A = self.venv.single_action_space.shape[0]
        self.low = torch.as_tensor(self.venv.single_action_space.low, device=self.dev)
        self.high = torch.as_tensor(self.venv.single_action_space.high, device=self.dev)
        self.T = n_ep * EP_LEN
        self.full_obs = full_obs  # True: actor sees the critic obs incl. the true hole (oracle baseline, use with n_ep=1)
        self.critic_dim = S_DIM + n_ep + 1 + 15 + 43
        self.actor_dim = self.critic_dim if full_obs else S_DIM + n_ep + 1 + 15

    # ---- meta-episode control
    def reset(self, seed, goals=None, gen=None):
        """Start a meta-episode. goals (N,3) box poses, default sampled from the split ('all'). Returns (actor obs, critic obs)."""
        self.goals = goals if goals is not None else checker_sample(self.N, self.train_region, self.dev, gen, self.split)
        self.seed, self.ep, self.t = seed, 0, 0
        self.f = torch.zeros(self.N, 7, device=self.dev)
        self.c = torch.zeros(self.N, 7, device=self.dev)
        self.fb_valid = torch.zeros(self.N, device=self.dev)
        self.succ = torch.zeros(self.N, self.n_ep, dtype=torch.bool, device=self.dev)
        self.f_hist, self.c_hist = [], []
        self._reset_episode()
        return self.obs()

    def _reset_episode(self):
        s = self.seed if self.same_init else self.seed + 7919 * self.ep
        obs, _ = self.venv.reset(seed=s, options=dict(box_pose=self.goals, offset=torch.zeros_like(self.goals)))
        self.raw = obs
        self.done_ep = torch.zeros(self.N, dtype=torch.bool, device=self.dev)  # first success reached in this episode
        self.f_cur = obs[:, PEG].clone()

    def obs(self):
        o = self.raw
        oh = torch.zeros(self.N, self.n_ep, device=self.dev)
        oh[:, self.ep] = 1
        tt = torch.full((self.N, 1), self.t / EP_LEN, device=self.dev)
        a = torch.cat([o[:, :S_DIM], oh, tt, torch.zeros_like(self.f) if self.hide_f else self.f, self.c, self.fb_valid[:, None]], 1)
        cr = torch.cat([a, o[:, :43]], 1)
        return (cr if self.full_obs else a), cr

    def expert_obs(self):
        return self.raw[:, :43]

    def step(self, action):
        """Returns (actor obs, critic obs, reward (N,), meta_done bool, info). info['ep_end'] is True on the last step of an episode."""
        action = torch.max(torch.min(action, self.high), self.low)
        obs, rew, _, _, info = self.venv.step(action)
        self.raw = obs
        s = info["success"].bool()
        newly = s & ~self.done_ep
        self.f_cur = torch.where((~self.done_ep)[:, None], obs[:, PEG], self.f_cur)  # frozen at the first success
        self.done_ep |= s
        self.succ[:, self.ep] |= newly
        self.t += 1
        ep_end = self.t == EP_LEN
        out_info = dict(ep_end=ep_end, ep=self.ep, success=s)
        if ep_end:
            hole = obs[:, HOLE]
            c = hole - self.f_cur
            if self.nxy or self.nyaw:
                c = c.clone()
                c[:, :2] += self.nxy * torch.randn(self.N, 2, device=self.dev)
                if self.nyaw:
                    q = quaternion_multiply(zrot(self.nyaw * torch.randn(self.N, device=self.dev)), hole[:, 3:])
                    c[:, 3:] = q - self.f_cur[:, 3:]
            self.f, self.c = self.f_cur.clone(), c
            self.fb_valid = torch.ones(self.N, device=self.dev)
            self.f_hist.append(self.f), self.c_hist.append(self.c)
            if self.ep + 1 < self.n_ep:
                self.ep += 1
                self.t = 0
                self._reset_episode()
                a, cr = self.obs()
                return a, cr, rew, False, out_info
            a, cr = self.obs()
            return a, cr, rew, True, out_info
        a, cr = self.obs()
        return a, cr, rew, False, out_info


def load_expert(path, device):
    from ppo import Agent

    class _S:
        single_observation_space = type("S", (), {"shape": (43,)})
        single_action_space = type("S", (), {"shape": (8,)})

    ex = Agent(_S).to(device)
    ex.load_state_dict(torch.load(path, map_location=device))
    ex.eval()
    return ex


def prior_hole_obs(o43):
    """Expert obs with the hole moved to the prior mean box pose (x 0, y 0.3, yaw pi/2), true height (icl_tasks.PegTask.prior_obs)."""
    o = o43.clone()
    o[:, 35:37] = torch.tensor([0.0, 0.3], device=o.device)
    o[:, 38:42] = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=o.device)
    return o


def believed_hole_obs(o43, hole_est):
    o = o43.clone()
    o[:, 35:42] = hole_est
    return o
