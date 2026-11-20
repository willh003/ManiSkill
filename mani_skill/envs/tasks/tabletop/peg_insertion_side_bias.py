import os

import numpy as np
import torch

from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.utils.geometry.rotation_conversions import quaternion_apply, quaternion_multiply
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

# PegInsertionSide-v1 box spawn ranges (see PegInsertionSideEnv._initialize_episode)
BOX_XY_LOW, BOX_XY_HIGH = (-0.05, 0.2), (0.05, 0.4)
BOX_YAW_LOW, BOX_YAW_HIGH = np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8


@register_env("PegInsertionSideBias-v1", max_episode_steps=100)
class PegInsertionSideBiasEnv(PegInsertionSideEnv):
    """
    PegInsertionSide-v1 with a hidden per-task hole pose.

    The state observation is PegInsertionSide-v1's (clean, privileged) followed by `hole_offset` (dx, dy, dyaw) as the last 3 entries.
    A student that must not know the hole pose either sees `box_hole_pose` shifted by the offset (`observed_hole_obs`, the "biased"
    task) or does not see it at all (the "hidden" task). Both are built from the options of `reset`, which keep the task fixed across
    the episodes of a task:
      offset:   (num_envs, 3) bias of the observed hole pose (dx, dy in m, dyaw in rad); default random in +-bias_xy / +-bias_yaw
      box_pose: (num_envs, 3) true box (x, y, yaw); default the PegInsertionSide-v1 random spawn
    """

    bias_xy = float(os.environ.get("PEG_OBS_BIAS_XY", 0.02))
    bias_yaw = float(os.environ.get("PEG_OBS_BIAS_YAW", 0.1))

    @staticmethod
    def make_fixed_table(box_poses, seed=0, shared_peg=False):
        """Fully specified tasks for `fixed_tasks`: row k = (hole pose from box_poses[k], peg dims, hole-centre offset, peg start pose, robot qpos noise).
        Everything is drawn once from the same distributions as PegInsertionSide-v1 and then frozen."""
        M = len(box_poses)
        rs = np.random.RandomState(1234 + seed)
        T = dict(box=box_poses.detach().cpu().float())
        T["len"] = torch.tensor(rs.uniform(0.085, 0.125, M), dtype=torch.float32)
        T["rad"] = torch.tensor(rs.uniform(0.015, 0.025, M), dtype=torch.float32)
        T["ctr"] = 0.5 * (T["len"] - T["rad"])[:, None] * torch.tensor(rs.uniform(-1, 1, (M, 2)), dtype=torch.float32)
        T["peg_xy"] = torch.tensor(rs.uniform([-0.1, -0.3], [0.1, 0.0], (M, 2)), dtype=torch.float32)
        T["peg_yaw"] = torch.tensor(rs.uniform(np.pi / 2 - np.pi / 3, np.pi / 2 + np.pi / 3, M), dtype=torch.float32)
        T["qnoise"] = torch.tensor(rs.normal(0, 0.02, (M, 9)), dtype=torch.float32)
        if shared_peg:
            # one peg (size + start pose) shared by all tasks, robot start noise stays random (resampled each reset): tasks differ only in the hole pose,
            # which is picked uniformly at random from box_poses at every reset -> the task cannot be identified from the initial observation
            for k in ("len", "rad", "ctr", "peg_xy", "peg_yaw"):
                T[k] = T[k][:1].expand(M, *T[k].shape[1:]).clone()
            del T["qnoise"]
            T["shared"] = torch.ones(1)
        return T

    def __init__(self, *args, task_pool=None, append_offset=True, fixed_tasks=None, near_hole_prob=0.0, near_hole_max_depth=0.1, **kwargs):
        # near_hole_prob: OmniReset-style reset bias: with this probability a (re)set starts with the peg already in the hole, pulled back by U(0.03, near_hole_max_depth) m
        # along the hole axis from the fully inserted pose (robot stays at its default pose, so it must still grasp and push). Training envs only.
        self.near_hole_prob = near_hole_prob
        self.near_hole_max_depth = near_hole_max_depth
        # fixed_tasks: table from make_fixed_table. Env i always runs task i % M exactly (hole, peg dims, peg start pose, robot noise): no randomness across resets.
        self.fixed_tasks = fixed_tasks
        # task_pool: (M, 3) box poses (x, y, yaw); when given and reset has no `box_pose` option, every (partial) reset draws a task uniformly from it.
        # append_offset=False: the state obs is PegInsertionSide-v1's 43-dim clean obs (no hole_offset).
        self.task_pool = task_pool
        self.append_offset = append_offset
        super().__init__(*args, **kwargs)

    def _sample_peg_dims(self):
        if self.fixed_tasks is None:
            return super()._sample_peg_dims()
        k = np.arange(self.num_envs) % len(self.fixed_tasks["len"])
        return self.fixed_tasks["len"].numpy()[k], self.fixed_tasks["rad"].numpy()[k], self.fixed_tasks["ctr"].numpy()[k]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        options = options or {}
        if self.fixed_tasks is not None:
            T = {k: v.to(self.device) for k, v in self.fixed_tasks.items()}
            kk = env_idx.to(self.device) % len(T["len"])
            b = len(env_idx)
            pos = torch.zeros(b, 3, device=self.device)
            pos[:, :2] = T["peg_xy"][kk]
            pos[:, 2] = self.peg_half_sizes[env_idx, 2]
            yaw = T["peg_yaw"][kk]
            quat = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], dim=1)
            self.peg.set_pose(Pose.create_from_pq(pos, quat))
            if "qnoise" in T:
                q0 = torch.tensor([0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04], device=self.device)
                qpos = q0 + T["qnoise"][kk]
                qpos[:, -2:] = 0.04
                self.agent.robot.set_qpos(qpos)
            if options.get("box_pose") is None:
                rows = torch.randint(len(T["box"]), (b,), device=self.device) if "shared" in T else kk
                options = dict(options, box_pose=T["box"][rows])
        if not hasattr(self, "hole_offset"):
            self.hole_offset = torch.zeros(self.num_envs, 3, device=self.device)
        offset = options.get("offset")
        if offset is None:
            offset = (torch.rand(len(env_idx), 3, device=self.device) * 2 - 1) * torch.tensor(
                [self.bias_xy, self.bias_xy, self.bias_yaw], device=self.device
            )
        else:
            offset = offset.to(self.device)
            offset = offset[env_idx] if len(offset) == self.num_envs else offset
        self.hole_offset[env_idx] = offset

        box_pose = options.get("box_pose")
        if box_pose is None and self.task_pool is not None:
            pool = self.task_pool.to(self.device)
            box_pose = pool[torch.randint(len(pool), (len(env_idx),), device=self.device)]
        if box_pose is not None:
            box_pose = box_pose.to(self.device)
            box_pose = box_pose[env_idx] if len(box_pose) == self.num_envs else box_pose
            pos = torch.zeros(len(env_idx), 3, device=self.device)
            pos[:, :2] = box_pose[:, :2]
            pos[:, 2] = self.peg_half_sizes[env_idx, 0]
            yaw = box_pose[:, 2]
            quat = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], dim=1)
            self.box.set_pose(Pose.create_from_pq(pos, quat))
        if self.near_hole_prob > 0:
            b = len(env_idx)
            m = torch.rand(b, device=self.device) < self.near_hole_prob
            depth = 0.03 + torch.rand(b, device=self.device) * (self.near_hole_max_depth - 0.03)
            back = Pose.create_from_pq(p=torch.stack([-depth, torch.zeros_like(depth), torch.zeros_like(depth)], dim=1))
            tgt = self.goal_pose[env_idx] * back
            cur = self.peg.pose[env_idx]
            self.peg.set_pose(Pose.create_from_pq(torch.where(m[:, None], tgt.p, cur.p), torch.where(m[:, None], tgt.q, cur.q)))

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        if self.obs_mode_struct.use_state and self.append_offset:
            obs["hole_offset"] = self.hole_offset
        return obs


# layout of the flattened state obs: qpos 0:9, qvel 9:18, tcp_pose 18:25, peg_pose 25:32, peg_half_size 32:35,
# box_hole_pose 35:42 (pos 35:38, quat wxyz 38:42), box_hole_radius 42, hole_offset 43:46
HOLE_SLICE = slice(35, 42)
OFFSET_SLICE = slice(43, 46)


def observed_hole_obs(obs):
    """Clean obs (..., 46) -> (..., 43) with box_hole_pose shifted by the offset: position + (dx, dy, 0), orientation rotated by dyaw about z."""
    o = obs[..., OFFSET_SLICE]
    out = obs[..., :43].clone()
    out[..., 35:37] += o[..., :2]
    half = o[..., 2] / 2
    qz = torch.stack([half.cos(), torch.zeros_like(half), torch.zeros_like(half), half.sin()], dim=-1)
    out[..., 38:42] = quaternion_multiply(qz, obs[..., 38:42])
    return out


def peg_head_pos(obs):
    """World position of the peg head (the end that is inserted) from a clean obs."""
    return obs[..., 25:28] + quaternion_apply(obs[..., 28:32], torch.cat([obs[..., 32:33], torch.zeros_like(obs[..., 33:35])], dim=-1))
