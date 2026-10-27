"""Allegro in-hand cube rotation where the (signed) rotation axis is a per-task variable.

Same scene as RotateSingleObjectInHandLevel0 (cube on an upward allegro palm). The task is to spin the cube by `ROT_SUCCESS_ANGLE`
radians (default pi) about a signed axis (+-x, +-y, +-z), the sign included (rotating the wrong way scores negatively).
The env obs always contains the true axis (`rotate_dir`, 3) and the signed cumulative angle; the student's task class hides them.

Reset option `rot_dir` (n, 3): signed one-hot axis. Otherwise sampled uniformly from the axes in env var `ROT_AXES` (default "xyz"),
both signs ("z" gives only +-z, "z+" only +z).
"""
import os

import torch
from mani_skill.envs.tasks.dexterity.rotate_single_object_in_hand import RotateSingleObjectInHand, RotateSingleObjectInHandLevel0
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import vectorize_pose
from mani_skill.utils.geometry.rotation_conversions import quaternion_apply


def allowed_dirs(spec: str):
    dirs = []
    for ax, ch in enumerate("xyz"):
        for sign, s in ((1.0, "+"), (-1.0, "-")):
            if ch in spec and ((s in spec) or not ("+" in spec or "-" in spec)):
                v = [0.0] * 3
                v[ax] = sign
                dirs.append(v)
    return dirs


@register_env("RotateDirHidden-v1", max_episode_steps=200)
class RotateDirHiddenEnv(RotateSingleObjectInHandLevel0):
    def __init__(self, *args, **kwargs):
        self.dirs = torch.tensor(allowed_dirs(os.environ.get("ROT_AXES", "xyz")))
        self.rot_success_angle = float(os.environ.get("ROT_SUCCESS_ANGLE", torch.pi))
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        b = len(env_idx)
        if options is not None and "rot_dir" in options:
            d = torch.as_tensor(options["rot_dir"], dtype=torch.float32, device=self.device).reshape(-1, 3)
            d = d.expand(b, 3) if len(d) == 1 else d
        else:
            d = self.dirs.to(self.device)[torch.randint(0, len(self.dirs), (b,), device=self.device)]
        if not hasattr(self, "sdir"):
            self.sdir = torch.zeros(self.num_envs, 3, device=self.device)
        self.sdir[env_idx] = d  # signed axis (the base class's rot_dir is its unsigned long one-hot)
        # tangent vector: any unit vector orthogonal to the axis
        axis = d.abs().argmax(dim=-1)
        self.unit_vector[env_idx] = torch.nn.functional.one_hot((axis + 1) % 3, 3).float()
        self.prev_unit_vector[env_idx] = self.unit_vector[env_idx].clone()
        self.cum_rotation_angle[env_idx] = 0.0
        self.success_threshold = self.rot_success_angle

    def _get_obs_extra(self, info):
        obs = super()._get_obs_extra(info)
        obs["rotate_dir"] = self.sdir
        obs["cum_angle"] = self.cum_rotation_angle[:, None]
        return obs

    def evaluate(self, **kwargs):
        with torch.device(self.device):
            q = self.obj.pose.q
            new_vec = quaternion_apply(q, self.unit_vector)
            new_vec = new_vec - (new_vec * self.sdir).sum(-1, keepdim=True) * self.sdir
            new_vec = new_vec / torch.linalg.norm(new_vec, dim=-1, keepdim=True).clamp_min(1e-6)
            sin = (torch.cross(self.prev_unit_vector, new_vec, dim=-1) * self.sdir).sum(-1)
            cos = (self.prev_unit_vector * new_vec).sum(-1)
            angle = torch.atan2(sin, cos).clip(-torch.pi / 20, torch.pi / 20)
            self.prev_unit_vector = new_vec
            self.cum_rotation_angle += angle
            success = self.cum_rotation_angle > self.success_threshold
        # reuse the rest of the base evaluate for velocity / fall / effort terms without touching the rotation state
        keep = (self.prev_unit_vector.clone(), self.cum_rotation_angle.clone())
        base = RotateSingleObjectInHand.evaluate(self, **kwargs)  # velocity / fall / effort terms; its unsigned rotation state is restored
        self.prev_unit_vector, self.cum_rotation_angle = keep
        base.update(rotation_angle=angle, success=success)
        return base
