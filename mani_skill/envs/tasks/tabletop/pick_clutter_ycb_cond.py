"""PickClutterYCBCond-v1: lift a conditioned target out of a fixed set of K YCB objects.

Every sub-scene holds the same K objects (`MODEL_IDS`), placed at random non-overlapping (x, y, yaw) on the table. The task is
"lift object `target`". The target identity is given to the policy only through the *target block* at the end of the state obs
(`TARGET_SLICE`), so a student can hide the condition by zeroing that block.

Success: target grasped and lifted >= LIFT_SUCCESS above its rest height.
Fail ("wrong object"): any non-target object lifted >= LIFT_WRONG. Both terminate the episode.
info["lifted_id"]: the lifted object (target if success, else the highest lifted non-target), -1 if none.

Reset options (keep a task fixed across attempts):
  layout: (n, K, 3) per-object (x, y, yaw); default random non-overlapping
  target: (n,) long in [0, K); default uniform
With `robot_init_qpos_noise=0` the initial state is fully determined by (layout, target).

Flattened state obs (panda, K=4), see `obs_layout()`:
  qpos 0:9, qvel 9:18, tcp_pose 18:25, obj_pose 25:53 (K x 7, by object id), obj_lift 53:57, obj_grasped 57:61,
  target block 61:75 = target_onehot 61:65, target_pose 65:72, tcp_to_target 72:75
"""

from typing import Any

import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

MODEL_IDS = ["077_rubiks_cube", "061_foam_brick", "011_banana", "056_tennis_ball"]
K = len(MODEL_IDS)
LIFT_SUCCESS = 0.10
LIFT_WRONG = 0.05
XY_LOW, XY_HIGH = (-0.15, -0.22), (0.15, 0.22)
CLEARANCE = 0.015  # extra gap between bounding circles


def obs_layout(k: int = K):
    """Slices of the flattened state obs."""
    o = 18
    lay = dict(qpos=slice(0, 9), qvel=slice(9, 18), tcp_pose=slice(o, o + 7))
    o += 7
    lay["obj_pose"] = slice(o, o + 7 * k); o += 7 * k
    lay["obj_lift"] = slice(o, o + k); o += k
    lay["obj_grasped"] = slice(o, o + k); o += k
    lay["target"] = slice(o, o + k + 10)
    lay["target_onehot"] = slice(o, o + k)
    lay["target_pose"] = slice(o + k, o + k + 7)
    lay["tcp_to_target"] = slice(o + k + 7, o + k + 10)
    lay["dim"] = o + k + 10
    return lay


TARGET_SLICE = obs_layout()["target"]
OBS_DIM = obs_layout()["dim"]


@register_env("PickClutterYCBCond-v1", max_episode_steps=75, asset_download_ids=["ycb"])
class PickClutterYCBCondEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    SUPPORTED_REWARD_MODES = ["normalized_dense", "dense", "sparse", "none"]
    agent: Panda

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        info = load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json")
        bb = [info[m]["bbox"] for m in MODEL_IDS]
        # rest height (bottom on the table) and xy bounding-circle radius per object
        self._rest_z = [-b["min"][2] for b in bb]
        self._radius = [float(np.hypot(b["max"][0], b["max"][1])) for b in bb]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(gpu_memory_config=GPUMemoryConfig(max_rigid_contact_count=2**21, max_rigid_patch_count=2**19))

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.objs = []
        for i, m in enumerate(MODEL_IDS):
            b = actors.get_actor_builder(self.scene, id=f"ycb:{m}")
            b.initial_pose = sapien.Pose(p=[0.3 * i, 1.0, self._rest_z[i]])
            self.objs.append(b.build(name=f"obj{i}-{m}"))

    # ---------------------------------------------------------------- episode init
    def sample_layout(self, n: int, gen: torch.Generator = None) -> torch.Tensor:
        """(n, K, 3) random (x, y, yaw) with non-overlapping bounding circles (rejection sampling, 128 candidates/object)."""
        dev = self.device
        M = 128
        lo, hi = torch.tensor(XY_LOW, device=dev), torch.tensor(XY_HIGH, device=dev)
        rad = torch.tensor(self._radius, device=dev)
        xy = torch.zeros(n, K, 2, device=dev)
        for k in range(K):
            cand = lo + torch.rand(n, M, 2, generator=gen, device=dev) * (hi - lo)
            if k == 0:
                xy[:, 0] = cand[:, 0]
                continue
            d = torch.linalg.norm(cand[:, :, None] - xy[:, None, :k], dim=-1)  # (n, M, k)
            ok = (d >= rad[k] + rad[:k] + CLEARANCE).all(-1)  # (n, M)
            idx = torch.where(ok.any(1), ok.float().argmax(1), torch.zeros(n, dtype=torch.long, device=dev))
            xy[:, k] = cand[torch.arange(n, device=dev), idx]
        yaw = (torch.rand(n, K, generator=gen, device=dev) * 2 - 1) * np.pi
        return torch.cat([xy, yaw[..., None]], -1)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        options = options or {}
        b = len(env_idx)
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            if not hasattr(self, "target"):
                self.target = torch.zeros(self.num_envs, dtype=torch.long)
                self.layout = torch.zeros(self.num_envs, K, 3)
                self.rest_z = torch.tensor(self._rest_z)

            def pick(v):
                v = torch.as_tensor(v, device=self.device)
                return v[env_idx] if len(v) == self.num_envs and b != self.num_envs else v

            layout = options.get("layout")
            layout = self.sample_layout(b) if layout is None else pick(layout).float()
            target = options.get("target")
            target = torch.randint(0, K, (b,)) if target is None else pick(target).long()
            self.layout[env_idx] = layout
            self.target[env_idx] = target
            for k, obj in enumerate(self.objs):
                p = torch.zeros(b, 3)
                p[:, :2] = layout[:, k, :2]
                p[:, 2] = self._rest_z[k]
                yaw = layout[:, k, 2]
                q = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], 1)
                obj.set_pose(Pose.create_from_pq(p, q))
                obj.set_linear_velocity(torch.zeros(b, 3))
                obj.set_angular_velocity(torch.zeros(b, 3))
            qpos = np.array([0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04])
            qpos = torch.tensor(qpos, dtype=torch.float32).repeat(b, 1)
            if self.robot_init_qpos_noise > 0:
                qpos[:, :-2] += torch.randn(b, 7) * self.robot_init_qpos_noise
            self.agent.reset(qpos)
            self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))

    # ---------------------------------------------------------------- task state
    def _obj_p(self):
        return torch.stack([o.pose.p for o in self.objs], 1)  # (n, K, 3)

    def _obj_raw(self):
        return torch.stack([o.pose.raw_pose for o in self.objs], 1)  # (n, K, 7)

    def evaluate(self):
        p = self._obj_p()
        lift = p[..., 2] - self.rest_z  # (n, K)
        grasped = torch.stack([self.agent.is_grasping(o) for o in self.objs], 1)  # (n, K)
        ar = torch.arange(self.num_envs, device=self.device)
        tmask = torch.nn.functional.one_hot(self.target, K).bool()
        t_lift, t_grasp = lift[ar, self.target], grasped[ar, self.target]
        success = (t_lift >= LIFT_SUCCESS) & t_grasp
        other_lift = lift.masked_fill(tmask, -1.0)
        fail = (other_lift >= LIFT_WRONG).any(1) & ~success
        best_other = other_lift.argmax(1)
        lifted_id = torch.where(success, self.target, torch.where(fail, best_other, torch.full_like(self.target, -1)))
        return dict(success=success, fail=fail, lifted_id=lifted_id, obj_lift=lift, obj_grasped=grasped,
                    is_grasped=t_grasp, target_lift=t_lift, target=self.target.clone())

    def _get_obs_extra(self, info: dict):
        ar = torch.arange(self.num_envs, device=self.device)
        tcp = self.agent.tcp.pose
        obs = dict(tcp_pose=tcp.raw_pose)
        if self.obs_mode_struct.use_state:
            raw = self._obj_raw()
            tp = raw[ar, self.target]
            obs.update(
                obj_pose=raw.reshape(self.num_envs, -1),
                obj_lift=info["obj_lift"],
                obj_grasped=info["obj_grasped"].float(),
                target_onehot=torch.nn.functional.one_hot(self.target, K).float(),
                target_pose=tp,
                tcp_to_target=tp[:, :3] - tcp.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        ar = torch.arange(self.num_envs, device=self.device)
        tp = self._obj_p()[ar, self.target]
        d = torch.linalg.norm(tp - self.agent.tcp.pose.p, dim=1)
        reward = 1 - torch.tanh(5 * d)
        g = info["is_grasped"].float()
        reward = reward + g
        reward = reward + 2 * g * (info["target_lift"] / LIFT_SUCCESS).clamp(0, 1)
        tmask = torch.nn.functional.one_hot(self.target, K).bool()
        wrong = (info["obj_lift"].masked_fill(tmask, 0) / LIFT_WRONG).clamp(0, 1).sum(1)
        wrong_grasp = info["obj_grasped"].masked_fill(tmask, False).any(1).float()
        reward = reward - wrong - 0.5 * wrong_grasp
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0
