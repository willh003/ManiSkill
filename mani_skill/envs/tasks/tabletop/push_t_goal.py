import os

import numpy as np
import torch

from mani_skill.envs.tasks.tabletop.push_t import PushTEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose


@register_env("PushTGoal-v1", max_episode_steps=100)
class PushTGoalEnv(PushTEnv):
    """
    PushT-v1 with a per-episode randomized goal T pose (x, y, yaw), sampled around the default goal.

    The state observation additionally contains the goal pose (position + cos/sin of yaw), placed before the T pose so the
    last 7 observation entries are still the T pose. The goal can be fixed across episodes by passing
    `options=dict(goal=(num_envs, 3) tensor of [dx, dy, dyaw])` to reset, which is how a multi-episode "task" is built.
    """

    # +- metres / degrees around the default goal pose; overridable via PUSHT_GOAL_POS_RANGE / PUSHT_GOAL_YAW_DEG so PPO
    # sweeps don't need new CLI plumbing
    goal_pos_range = float(os.environ.get("PUSHT_GOAL_POS_RANGE", 0.03))
    goal_yaw_range = float(os.environ.get("PUSHT_GOAL_YAW_DEG", 45.0)) * np.pi / 180

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        # parent builds a single (3,3) world->goal transform; the intersection code broadcasts a (num_envs,3,3) one
        self.world_to_goal_trans = (
            self.world_to_goal_trans.unsqueeze(0).repeat(self.num_envs, 1, 1).clone()
        )
        self.goal_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.goal_yaw = torch.zeros(self.num_envs, device=self.device)
        self.goal_delta = torch.zeros(self.num_envs, 3, device=self.device)

    def sample_goal_delta(self, b):
        u = torch.rand(b, 3, device=self.device) * 2 - 1
        return u * torch.tensor(
            [self.goal_pos_range, self.goal_pos_range, self.goal_yaw_range],
            device=self.device,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            delta = options.get("goal") if options is not None else None
            if delta is None:
                delta = self.sample_goal_delta(b)
            else:
                delta = delta.to(self.device)[env_idx] if len(delta) == self.num_envs else delta.to(self.device)
            default_xy = self.goal_offset.to(self.device)
            goal_xy = default_xy + delta[:, :2]
            goal_yaw = (self.goal_z_rot + delta[:, 2]) % (2 * np.pi)
            self.goal_delta[env_idx] = delta
            self.goal_xy[env_idx] = goal_xy
            self.goal_yaw[env_idx] = goal_yaw

            # world->goal 2D homogeneous transform per env
            c, s = goal_yaw.cos(), goal_yaw.sin()
            goal_trans = torch.eye(3).repeat(b, 1, 1)
            goal_trans[:, 0, 0] = c
            goal_trans[:, 0, 1] = -s
            goal_trans[:, 1, 0] = s
            goal_trans[:, 1, 1] = c
            goal_trans[:, :2, 2] = goal_xy
            self.world_to_goal_trans[env_idx] = torch.linalg.inv(goal_trans)

            goal_p = torch.zeros((b, 3))
            goal_p[:, :2] = goal_xy
            goal_p[:, 2] = 1e-3
            goal_q = torch.zeros((b, 4))
            goal_q[:, 0] = (goal_yaw / 2).cos()
            goal_q[:, 3] = (goal_yaw / 2).sin()
            self.goal_tee.set_pose(Pose.create_from_pq(p=goal_p, q=goal_q))

            # T spawn: same box as PushT-v1, relative to the *default* goal so the start distribution doesn't move with the goal
            p = torch.zeros((b, 3))
            p[:, :2] = default_xy
            p[:, 0] += torch.rand(b) * self.tee_spawnbox_xlength + self.tee_spawnbox_xoffset
            p[:, 1] += torch.rand(b) * self.tee_spawnbox_ylength + self.tee_spawnbox_yoffset
            p[:, 2] = 0.04 / 2 + 1e-3
            yaw = torch.rand(b) * (2 * torch.pi)
            q = torch.zeros((b, 4))
            q[:, 0] = (yaw / 2).cos()
            q[:, -1] = (yaw / 2).sin()
            self.tee.set_pose(Pose.create_from_pq(p=p, q=q))

            xyz = torch.zeros((b, 3))
            xyz[:] = self.ee_starting_pos2D
            self.ee_goal_pos.set_pose(
                Pose.create_from_pq(p=xyz, q=torch.tensor([0.7071068, 0, 0.7071068, 0]).repeat(b, 1))
            )

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                goal_pos=self.goal_tee.pose.p,
                goal_cs=torch.stack([self.goal_yaw.cos(), self.goal_yaw.sin()], dim=1),
                obj_pose=self.tee.pose.raw_pose,
            )
        return obs

    def tee_goal_error(self):
        """Privileged [dx, dy, cos(dyaw), sin(dyaw)] of the T relative to the goal (zero error = [0, 0, 1, 0])."""
        dxy = self.tee.pose.p[:, :2] - self.goal_xy
        dyaw = self.quat_to_z_euler(self.tee.pose.q) - self.goal_yaw
        return torch.cat([dxy, dyaw.cos()[:, None], dyaw.sin()[:, None]], dim=1)

    def compute_dense_reward(self, obs, action, info):
        tee_z_eulers = self.quat_to_z_euler(self.tee.pose.q)
        rot_rew = (tee_z_eulers - self.goal_yaw).cos()
        reward = (((rot_rew + 1) / 2) ** 2) / 2

        tee_to_goal_dist = torch.linalg.norm(self.tee.pose.p[:, :2] - self.goal_xy, dim=1)
        reward += ((1 - torch.tanh(5 * tee_to_goal_dist)) ** 2) / 2

        tcp_to_push_dist = torch.linalg.norm(self.tee.pose.p - self.agent.tcp.pose.p, dim=1)
        reward += ((1 - torch.tanh(5 * tcp_to_push_dist)).sqrt()) / 20

        reward[info["success"]] = 3
        return reward
