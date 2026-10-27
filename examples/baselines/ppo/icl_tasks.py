"""Task definitions for icl_episodic.py: what is hidden from the student, what the expert sees, and the privileged final-state feedback.

A task class provides
  obs_dim, y_dim          size of the env's state obs and of the `final_error` feedback vector
  sample(base, n)         n task variables (the hidden quantity, fixed across a task's k episodes)
  reset_options(tasks)    env reset options that instantiate them
  student_obs(obs)        the observation the student gets (hidden entries removed / biased)
  expert_obs(obs)         the clean observation the expert gets
  prior_obs(obs, base)    obs for an "expert told the default task", the best a context-free policy could do
  final_error(obs)        feedback computed from the privileged final obs
  metrics(err)            (position error, yaw error) of a feedback vector, for logging
To add a task: write a class here and register it in TASKS.
"""
import torch


def yaw_of_quat(q):
    return 2 * torch.atan2(q[..., 3], q[..., 0])


# observation layout of PushTGoal-v1 (33 dims): qpos 0:7, qvel 7:14, tcp pose 14:21, goal pos 21:24, goal cos/sin 24:26, T pose 26:33
GOAL_SLICE = slice(21, 26)


class PushTTask:
    """Hidden goal T pose. The student sees the clean obs without the goal entries."""

    obs_dim, y_dim = 33, 4

    def sample(self, base, n):
        return base.sample_goal_delta(n)

    def reset_options(self, tasks):
        return dict(goal=tasks)

    def student_obs(self, obs):
        return torch.cat([obs[..., : GOAL_SLICE.start], obs[..., GOAL_SLICE.stop :]], dim=-1)

    def expert_obs(self, obs):
        return obs


    def prior_obs(self, obs, base):
        """Obs for an expert that is told the default goal."""
        o = obs.clone()
        o[:, GOAL_SLICE.start : GOAL_SLICE.start + 2] = torch.tensor(base.goal_offset.tolist(), device=obs.device)
        o[:, 23] = 1e-3
        o[:, 24] = torch.cos(torch.tensor(base.goal_z_rot, device=obs.device))
        o[:, 25] = torch.sin(torch.tensor(base.goal_z_rot, device=obs.device))
        return o

    def final_error(self, obs):
        """Final T pose relative to the goal, [dx, dy, cos dyaw, sin dyaw]."""
        dxy = obs[:, 26:28] - obs[:, 21:23]
        goal_yaw = torch.atan2(obs[:, 25], obs[:, 24])
        dyaw = yaw_of_quat(obs[:, 29:33]) - goal_yaw
        return torch.cat([dxy, dyaw.cos()[:, None], dyaw.sin()[:, None]], dim=1)

    def metrics(self, err):
        return err[:, :2].norm(dim=1), torch.atan2(err[:, 3], err[:, 2]).abs()


class PickBiasTask:
    """Hidden xy bias between the observed and the true cube position (PickCubeBias-v1). The student sees the biased cube
    entries; the expert sees the clean obs. The feedback is the true tcp -> cube vector at the end of the episode."""

    obs_dim, y_dim = 45, 3

    def sample(self, base, n):
        o = torch.zeros(n, 3, device=base.device)
        o[:, :2] = (torch.rand(n, 2, device=base.device) * 2 - 1) * base.bias_range
        return o

    def reset_options(self, tasks):
        return dict(offset=tasks)

    def student_obs(self, obs):
        from mani_skill.envs.tasks.tabletop.pick_cube_bias import observed_cube_obs

        return observed_cube_obs(obs)[..., :42]

    def expert_obs(self, obs):
        return obs[..., :42]

    def prior_obs(self, obs, base):
        """An expert that assumes there is no bias acts on the biased observation."""
        from mani_skill.envs.tasks.tabletop.pick_cube_bias import observed_cube_obs

        return observed_cube_obs(obs)

    def final_error(self, obs):
        return obs[:, 36:39]

    def metrics(self, err):
        return err[:, :2].norm(dim=1), torch.zeros(len(err), device=err.device)


class PegTask:
    """Hidden hole pose (PegInsertionSideBias-v1). hide_hole=False: the student sees box_hole_pose shifted by a per-task offset
    (dx, dy, dyaw); hide_hole=True: the student does not see it, and the task is the true box pose (x, y, yaw), fixed across the
    task's episodes. The expert sees the clean obs. The feedback is the true hole pose relative to the final peg head:
    [hole - head (world frame, 3), head (3), cos(hole yaw), sin(hole yaw)], so the hole pose is a function of the token."""

    obs_dim, y_dim = 46, 8

    def __init__(self, hide_hole=False, full_info=False):
        self.hide_hole = hide_hole
        self.full_info = full_info  # the student sees the true hole pose; the task is still a fixed box pose per task
        self.keep = [i for i in range(42) if not (hide_hole and 35 <= i < 42)]  # 42 = hole radius = peg radius + 3 mm: redundant

    def sample(self, base, n):
        from mani_skill.envs.tasks.tabletop import peg_insertion_side_bias as pb

        d = base.device
        if self.hide_hole or self.full_info:
            lo = torch.tensor([*pb.BOX_XY_LOW, pb.BOX_YAW_LOW], device=d)
            hi = torch.tensor([*pb.BOX_XY_HIGH, pb.BOX_YAW_HIGH], device=d)
            return lo + (hi - lo) * torch.rand(n, 3, device=d)
        return (torch.rand(n, 3, device=d) * 2 - 1) * torch.tensor([base.bias_xy, base.bias_xy, base.bias_yaw], device=d)

    def reset_options(self, tasks):
        if self.hide_hole or self.full_info:
            return dict(box_pose=tasks, offset=torch.zeros_like(tasks))
        return dict(offset=tasks)

    def student_obs(self, obs):
        from mani_skill.envs.tasks.tabletop.peg_insertion_side_bias import observed_hole_obs

        return observed_hole_obs(obs)[..., self.keep]

    def expert_obs(self, obs):
        return obs[..., :43]


    def prior_obs(self, obs, base):
        """Hidden: an expert told the hole is at the default box pose (true height). Biased: an expert that assumes there is no bias."""
        from mani_skill.envs.tasks.tabletop.peg_insertion_side_bias import observed_hole_obs

        o = observed_hole_obs(obs)
        if self.hide_hole:
            o = obs[..., :43].clone()
            o[..., 35:37] = torch.tensor([0.0, 0.3], device=obs.device)
            o[..., 38:42] = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=obs.device)
        return o

    def final_error(self, obs):
        from mani_skill.envs.tasks.tabletop.peg_insertion_side_bias import peg_head_pos

        head = peg_head_pos(obs)
        yaw = yaw_of_quat(obs[..., 38:42])
        return torch.cat([obs[..., 35:38] - head, head, yaw.cos()[..., None], yaw.sin()[..., None]], dim=-1)

    def metrics(self, err):
        return err[:, :3].norm(dim=1), torch.zeros(len(err), device=err.device)


class RotTask:
    """Hidden signed rotation axis (RotateDirHidden-v1, allegro hand): the task is the true axis (+-x, +-y, +-z), fixed across the task's
    episodes. The student does not see the axis (obs 83:86) nor the cumulative angle (105); the expert sees the full 106-dim obs.
    The feedback is [true axis (3), signed cumulative rotation (rad)] at the end of the episode."""

    obs_dim, y_dim = 106, 4
    DIR = slice(83, 86)
    CUM = 105

    def __init__(self):
        self.keep = [i for i in range(106) if not (83 <= i < 86 or i == 105)]

    def sample(self, base, n):
        return base.dirs.to(base.device)[torch.randint(0, len(base.dirs), (n,), device=base.device)]

    def reset_options(self, tasks):
        return dict(rot_dir=tasks)

    def student_obs(self, obs):
        return obs[..., self.keep]

    def expert_obs(self, obs):
        return obs


    def prior_obs(self, obs, base):
        """An expert told the first allowed axis (default +z)."""
        o = obs.clone()
        o[..., self.DIR] = base.dirs[0].to(obs.device)
        return o

    def final_error(self, obs):
        return torch.cat([obs[..., self.DIR], obs[..., self.CUM : self.CUM + 1]], dim=-1)

    def metrics(self, err):
        return err[:, 3].abs(), torch.zeros(len(err), device=err.device)


TASKS = {"PushTGoal-v1": PushTTask, "PickCubeBias-v1": PickBiasTask, "PegInsertionSideBias-v1": PegTask, "RotateDirHidden-v1": RotTask}
