import os

import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("PickCubeBias-v1", max_episode_steps=50)
class PickCubeBiasEnv(PickCubeEnv):
    """
    PickCube-v1 with a per-task bias between where the cube is observed and where it really is.

    The state observation is PickCube-v1's (clean, privileged) followed by `cube_offset` (xyz, z = 0) as the last 3 entries.
    A student that must not know the offset recomputes the cube-dependent entries as observed = true + offset
    (obj_pose[:3] + o, tcp_to_obj_pos + o, obj_to_goal_pos - o) from the clean obs; see `observed_cube_obs`. The offset is fixed
    across episodes by passing `options=dict(offset=(num_envs, 3) tensor)` to reset, which is how a multi-episode "task" is built.
    """

    # uniform +- metres in x and y; overridable via PICK_CUBE_OBS_BIAS
    bias_range = float(os.environ.get("PICK_CUBE_OBS_BIAS", 0.05))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if not hasattr(self, "cube_offset"):
            self.cube_offset = torch.zeros(self.num_envs, 3, device=self.device)
        offset = options.get("offset") if options is not None else None
        if offset is None:
            offset = torch.zeros(len(env_idx), 3, device=self.device)
            offset[:, :2] = (torch.rand(len(env_idx), 2, device=self.device) * 2 - 1) * self.bias_range
        else:
            offset = offset.to(self.device)
            offset = offset[env_idx] if len(offset) == self.num_envs else offset
        self.cube_offset[env_idx] = offset

        # fix the cube xy (a hidden per-task cube position, same for all episodes of a task); the yaw and the goal stay random
        cube_xy = options.get("cube_xy") if options is not None else None
        if cube_xy is not None:
            cube_xy = cube_xy.to(self.device)
            cube_xy = cube_xy[env_idx] if len(cube_xy) == self.num_envs else cube_xy
            p, q = self.cube.pose.p[env_idx].clone(), self.cube.pose.q[env_idx]
            p[:, :2] = cube_xy
            self.cube.set_pose(Pose.create_from_pq(p, q))

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        if "state" in self.obs_mode:
            obs["cube_offset"] = self.cube_offset
        return obs


# layout of the flattened state obs (panda): qpos 0:9, qvel 9:18, is_grasped 18, tcp_pose 19:26, goal_pos 26:29,
# obj_pose 29:36, tcp_to_obj_pos 36:39, obj_to_goal_pos 39:42, cube_offset 42:45
OFFSET_SLICE = slice(42, 45)


def observed_cube_obs(obs):
    """Clean obs (..., 45) -> what the student sees (..., 42): cube-dependent entries shifted by the offset, offset removed."""
    o = obs[..., OFFSET_SLICE]
    out = obs[..., : OFFSET_SLICE.start].clone()
    out[..., 29:32] += o
    out[..., 36:39] += o
    out[..., 39:42] -= o
    return out
