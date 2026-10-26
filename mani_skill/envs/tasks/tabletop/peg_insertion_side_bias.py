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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        options = options or {}
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
        if box_pose is not None:
            box_pose = box_pose.to(self.device)
            box_pose = box_pose[env_idx] if len(box_pose) == self.num_envs else box_pose
            pos = torch.zeros(len(env_idx), 3, device=self.device)
            pos[:, :2] = box_pose[:, :2]
            pos[:, 2] = self.peg_half_sizes[env_idx, 0]
            yaw = box_pose[:, 2]
            quat = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], dim=1)
            self.box.set_pose(Pose.create_from_pq(pos, quat))

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        if self.obs_mode_struct.use_state:
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
