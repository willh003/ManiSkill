"""Episodic in-context learning on PushTGoal-v1 (hidden goal pose, corrective feedback after each episode).

A "task" is a goal T pose the student cannot observe (the expert can). The student rolls out up to k episodes on the same
goal; after each episode i it receives feedback y_i, here the privileged final-state error of the T relative to the goal,
[dx, dy, cos(dyaw), sin(dyaw)]. Context c_h = ((xi_0, y_0), ..., (xi_{h-1}, y_{h-1})). The student is a transformer
pi(a | s, c_h): context tokens are subsampled (s, a) steps of past episodes plus one feedback token per episode; each
current-state query cross-attends to them (a learned null token is always present, so h = 0 works).

Episodes terminate at the first success, as in expert training (the env sets terminated = success). The vector env keeps
stepping, so everything after the first success is masked: it is excluded from the loss, from the context, and the feedback
is computed at the success step. The only success metric is therefore success_once, plus the episode length.

Each DAgger iteration:
  1. sample N tasks; roll out the student for k episodes, each conditioned on its own context so far (iteration 0, and a
     fraction of later episodes, use an expert that is told the default goal, so the context contains realistic failures)
  2. roll out the expert (true goal) for one episode on the same tasks
  3. label every visited state with the expert's action a*(s, goal). The expert episode is paired with every context c_h,
     h = 0..k; each student episode j is paired with its own context c_j (--no-relabel-student keeps only the expert pairs)
  4. aggregate, train by regression, evaluate the student alone on fixed fresh goals for k+1 episodes, so the curve
     success(h) shows whether more context helps. Controls: shuffled feedback, no feedback, no context.

Example:
    python icl_episodic.py --exp-name icl
"""
import time
from dataclasses import dataclass

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tyro

import mani_skill.envs  # noqa: F401  (registers PushTGoal-v1)
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from ppo import Agent
from run_logging import LOG_ROOT, RunDir

# observation layout of PushTGoal-v1 (33 dims): qpos 0:7, qvel 7:14, tcp pose 14:21, goal pos 21:24, goal cos/sin 24:26, T pose 26:33
GOAL_SLICE = slice(21, 26)
Y_DIM = 4


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

    def kin_idx(self):
        raise NotImplementedError("--ctx-features kin is only defined for PickCubeBias-v1")

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

    def __init__(self, hide_grasp="none", hide_cube=False):
        self.hide_cube = hide_cube
        # dims removed from the student's obs: the is_grasped flag (18), plus the finger joints that also reveal a grasp (qpos 7:9, qvel 16:18)
        drop = {"none": [], "flag": [18], "fingers": [7, 8, 16, 17, 18]}[hide_grasp]
        if hide_cube:  # the student gets no cube information at all: obj_pose 29:36, tcp_to_obj_pos 36:39, obj_to_goal_pos 39:42
            drop = drop + list(range(29, 42))
        self.keep = [i for i in range(42) if i not in drop]

    def sample(self, base, n):
        o = torch.zeros(n, 3, device=base.device)
        if self.hide_cube:  # the task is a hidden cube xy, fixed across the task's episodes
            o[:, :2] = (torch.rand(n, 2, device=base.device) * 2 - 1) * base.cube_spawn_half_size
            o[:, :2] += torch.tensor(base.cube_spawn_center, device=base.device)
            return o
        o[:, :2] = (torch.rand(n, 2, device=base.device) * 2 - 1) * base.bias_range
        return o

    def reset_options(self, tasks):
        if self.hide_cube:
            return dict(cube_xy=tasks[:, :2], offset=torch.zeros_like(tasks))
        return dict(offset=tasks)

    def student_obs(self, obs):
        from mani_skill.envs.tasks.tabletop.pick_cube_bias import observed_cube_obs

        return observed_cube_obs(obs)[..., self.keep]

    def expert_obs(self, obs):
        return obs[..., :42]

    def kin_idx(self):
        """Student-obs positions of tcp pose (19:26) and the observed cube xyz (29:32)."""
        return [self.keep.index(i) for i in list(range(19, 26)) + [29, 30, 31]]

    def prior_obs(self, obs, base):
        """An expert that assumes there is no bias acts on the biased observation."""
        from mani_skill.envs.tasks.tabletop.pick_cube_bias import observed_cube_obs

        if self.hide_cube:  # an expert told the cube is at the table centre
            o = obs[..., :42].clone()
            cube = torch.tensor([0.0, 0.0, 0.02], device=obs.device)
            o[..., 29:32] = cube
            o[..., 36:39] = cube - o[..., 19:22]
            o[..., 39:42] = o[..., 26:29] - cube
            return o
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

    def __init__(self, hide_hole=False):
        self.hide_hole = hide_hole
        self.keep = [i for i in range(42) if not (hide_hole and 35 <= i < 42)]  # 42 = hole radius = peg radius + 3 mm: redundant

    def sample(self, base, n):
        from mani_skill.envs.tasks.tabletop import peg_insertion_side_bias as pb

        d = base.device
        if self.hide_hole:
            lo = torch.tensor([*pb.BOX_XY_LOW, pb.BOX_YAW_LOW], device=d)
            hi = torch.tensor([*pb.BOX_XY_HIGH, pb.BOX_YAW_HIGH], device=d)
            return lo + (hi - lo) * torch.rand(n, 3, device=d)
        return (torch.rand(n, 3, device=d) * 2 - 1) * torch.tensor([base.bias_xy, base.bias_xy, base.bias_yaw], device=d)

    def reset_options(self, tasks):
        if self.hide_hole:
            return dict(box_pose=tasks, offset=torch.zeros_like(tasks))
        return dict(offset=tasks)

    def student_obs(self, obs):
        from mani_skill.envs.tasks.tabletop.peg_insertion_side_bias import observed_hole_obs

        return observed_hole_obs(obs)[..., self.keep]

    def expert_obs(self, obs):
        return obs[..., :43]

    def kin_idx(self):
        raise NotImplementedError("--ctx-features kin is only defined for PickCubeBias-v1")

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


TASKS = {"PushTGoal-v1": PushTTask, "PickCubeBias-v1": PickBiasTask, "PegInsertionSideBias-v1": PegTask}
TASK = PushTTask()


@dataclass
class Args:
    expert_checkpoint: str = "runs/pusht-goal-ppo/final_ckpt.pt"
    exp_name: str = "icl"
    out_dir: str = ""
    """default: <repo>/logs/runs/icl-<exp_name> (layout: see run_logging.py)"""
    env_id: str = "PushTGoal-v1"
    num_envs: int = 512
    """tasks (goals) per iteration and per evaluation; each is rolled out for k student episodes"""
    num_steps: int = 100
    k: int = 4
    """student episodes per task, i.e. context lengths h = 0..k"""
    num_iterations: int = 20
    hide_cube: bool = False
    """PickCubeBias only: the student gets no cube observation, and the task is the cube xy (fixed across a task's episodes)"""
    hide_hole: bool = False
    """PegInsertionSideBias only: the student gets no hole pose, and the task is the box pose (x, y, yaw) (else: biased observed hole pose)"""
    hide_grasp: str = "none"
    """PickCubeBias only: none | flag (drop is_grasped from the student obs) | fingers (also drop the finger joint qpos/qvel)"""
    ctx_features: str = "all"
    """features of the context (past-episode) steps: all | kin (PickCubeBias only: tcp pose + observed cube xyz, no joints/velocities/actions).
    The current-state query always sees the full obs."""
    feedback: str = "final_error"
    """final_error: [dx, dy, cos dyaw, sin dyaw] of the final T vs the goal; iou: scalar overlap of the final T with the goal
    (0..1); success: the 0/1 success flag; hints: no outcome feedback, instead n_hints (state, expert action) pairs per episode
    at the steps where the student's action differed most from the expert's"""
    n_hints: int = 5
    """number of hint steps per episode (--feedback hints)"""
    d_model: int = 128
    n_heads: int = 4
    n_ctx_layers: int = 2
    n_dec_layers: int = 2
    ctx_stride: int = 5
    """context tokens are every ctx_stride-th step of a past episode (always including the last step)"""
    grad_steps: int = 2000
    batch_size: int = 256
    lr: float = 3e-4
    prior_expert_frac: float = 0.25
    """fraction of student-episode slots (after iteration 0) driven by an expert told the default goal"""
    relabel_student: bool = True
    seed: int = 1
    eval_seed: int = 12345


def student_obs(obs):
    return TASK.student_obs(obs)


def yaw_of_quat(q):
    return 2 * torch.atan2(q[..., 3], q[..., 0])


def feedback_dim(args):
    return {"final_error": TASK.y_dim, "iou": 1, "success": 1, "hints": 1, "final_error_tcp": 6, "init_error_tcp": 6}[args.feedback]


def feedback_from_obs(obs, kind, success=None):
    """Feedback computed from the (privileged) final observation, which still contains the goal and the T pose."""
    if kind == "success":
        return success.float()[:, None]
    return TASK.final_error(obs)


class ICLPolicy(nn.Module):
    def __init__(self, s_dim, a_dim, y_dim, args):
        super().__init__()
        d, H = args.d_model, args.n_heads
        self.args = args
        self.register_buffer("s_mean", torch.zeros(s_dim))
        self.register_buffer("s_std", torch.ones(s_dim))
        self.register_buffer("y_mean", torch.zeros(y_dim))
        self.register_buffer("y_std", torch.ones(y_dim))
        self.step_in = nn.Linear(s_dim + a_dim, d)
        ctx_mask = torch.ones(s_dim + a_dim)
        if args.ctx_features == "kin":
            ctx_mask.zero_()
            ctx_mask[TASK.kin_idx()] = 1.0
        self.register_buffer("ctx_mask", ctx_mask, persistent=False)  # zeroes the context step features that are not used
        self.y_in = nn.Linear(y_dim, d)
        self.ep_emb = nn.Embedding(args.k, d)
        self.pos_emb = nn.Embedding(args.num_steps // args.ctx_stride + 1, d)
        self.type_emb = nn.Embedding(4, d)  # 0 null, 1 step, 2 feedback, 3 hint
        self.hint_in = nn.Linear(s_dim + a_dim, d)
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.ctx_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, H, 4 * d, dropout=0.0, batch_first=True, norm_first=True), args.n_ctx_layers, enable_nested_tensor=False
        )
        self.q_in = nn.Sequential(nn.Linear(s_dim, d), nn.ReLU(), nn.Linear(d, d))
        self.cross = nn.ModuleList([nn.MultiheadAttention(d, H, batch_first=True) for _ in range(args.n_dec_layers)])
        self.ln_q = nn.ModuleList([nn.LayerNorm(d) for _ in range(args.n_dec_layers)])
        self.mlp = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d)) for _ in range(args.n_dec_layers)]
        )
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, a_dim))

    def set_norm(self, s, y):
        s = s.reshape(-1, s.shape[-1])
        self.s_mean.copy_(s.mean(0))
        self.s_std.copy_(s.std(0).clamp_min(1e-3))
        y = y.reshape(-1, y.shape[-1])
        self.y_mean.copy_(y.mean(0))
        self.y_std.copy_(y.std(0).clamp_min(1e-3))

    def encode_context(self, ctx_steps, ctx_y, n_eps, ctx_valid, hints, hint_valid):
        """ctx_steps (B, k, Tc, S+A) raw, ctx_y (B, k, Y), n_eps (B,) number of valid past episodes,
        ctx_valid (B, k, Tc) False for steps after the episode terminated, hints (B, k, m, S+A) raw (state, expert action)
        pairs and hint_valid (B, k, m) (m = 0 when there are no hints).
        Returns memory (B, 1 + k*(Tc+1), d) and a key padding mask (True = ignore)."""
        B, k, Tc, _ = ctx_steps.shape
        dev = ctx_steps.device
        s = (ctx_steps[..., : self.s_mean.shape[0]] - self.s_mean) / self.s_std
        x = torch.cat([s, ctx_steps[..., self.s_mean.shape[0] :]], dim=-1) * self.ctx_mask
        ep = self.ep_emb(torch.arange(k, device=dev))[None, :, None]  # (1,k,1,d)
        pos = self.pos_emb(torch.arange(Tc, device=dev))[None, None]  # (1,1,Tc,d)
        step_tok = self.step_in(x) + ep + pos + self.type_emb.weight[1]
        y_tok = self.y_in((ctx_y - self.y_mean) / self.y_std)[:, :, None] + ep + self.type_emb.weight[2]
        h = torch.cat([(hints[..., : self.s_mean.shape[0]] - self.s_mean) / self.s_std, hints[..., self.s_mean.shape[0] :]], dim=-1)
        hint_tok = self.hint_in(h) + ep + self.type_emb.weight[3]
        m = hints.shape[2]
        tokens = torch.cat([step_tok, y_tok, hint_tok], dim=2).reshape(B, k * (Tc + 1 + m), -1)
        ep_valid = (torch.arange(k, device=dev)[None] < n_eps[:, None])[:, :, None]  # (B,k,1)
        valid = torch.cat([ctx_valid & ep_valid, ep_valid, hint_valid & ep_valid], dim=2).reshape(B, -1)
        null = self.null.expand(B, 1, -1) + self.type_emb.weight[0]
        tokens = torch.cat([null, tokens], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=dev), ~valid], dim=1)
        return self.ctx_enc(tokens, src_key_padding_mask=pad), pad

    def decode(self, s, memory, pad):
        """s (B, T, S) raw states -> actions (B, T, A)."""
        q = self.q_in((s - self.s_mean) / self.s_std)
        for attn, ln, mlp in zip(self.cross, self.ln_q, self.mlp):
            h = ln(q)
            q = q + attn(h, memory, memory, key_padding_mask=pad, need_weights=False)[0]
            q = q + mlp(q)
        return self.head(q)


def ctx_step_index(T, stride, device):
    """Indices of the steps kept as context tokens: every stride-th step, ending at the last step."""
    idx = torch.arange(stride - 1, T, stride, device=device)
    if idx[-1] != T - 1:
        idx = torch.cat([idx, torch.tensor([T - 1], device=device)])
    return idx


def rollout(venv, expert, policy, args, goals, seed, mode, ctx=None, ctx_mode="normal"):
    """One episode on every task. mode: 'student' | 'expert' | 'prior_expert' (expert told the default goal).
    ctx: dict(steps (N,k,Tc,S+A), y (N,k,Y), n (N,)) of previous episodes; ctx_mode: normal | shuf_y | no_y | no_ctx.
    Returns states, executed actions, expert labels, success flags and the feedback for this episode."""
    device, N, T = venv.device, args.num_envs, args.num_steps
    low = torch.from_numpy(venv.single_action_space.low).to(device)
    high = torch.from_numpy(venv.single_action_space.high).to(device)
    obs, _ = venv.reset(seed=seed, options=TASK.reset_options(goals))
    obs0 = obs.clone()  # clean obs at the start of the episode (PickCubeBias: includes the true cube position)
    if mode == "student":
        n, cy, ch, chv = ctx["n"], ctx["y"], ctx["hints"], ctx["hint_valid"]
        if ctx_mode == "shuf_y":  # feedback (and hints) of another task
            perm = torch.randperm(N, device=device)
            cy, ch, chv = cy[perm], ch[perm], chv[perm]
        elif ctx_mode == "no_y":  # keep the student's own episodes, drop all feedback
            cy, chv = torch.zeros_like(cy), torch.zeros_like(chv)
        elif ctx_mode == "no_ctx":
            n = torch.zeros_like(n)
        with torch.no_grad():
            memory, pad = policy.encode_context(ctx["steps"], cy, n, ctx["valid"], ch, chv)
    xs, acts, labels, alive = [], [], [], []
    succ_once = torch.zeros(N, dtype=torch.bool, device=device)  # also "done": the episode terminates at the first success
    y_obs = None  # observation at the termination step (the last step if it never succeeds)
    iou = None  # overlap at the termination step; the env auto-resets on the last step, so there it reuses the previous value
    base = venv.unwrapped
    for t in range(T):
        with torch.no_grad():
            a_lab = expert.get_action(TASK.expert_obs(obs), deterministic=True)
            if mode == "expert":
                a = a_lab
            elif mode == "prior_expert":
                a = expert.get_action(TASK.expert_obs(TASK.prior_obs(obs, base)), deterministic=True)
            else:
                a = policy.decode(student_obs(obs)[:, None], memory, pad)[:, 0]
        a, a_lab = torch.max(torch.min(a, high), low), torch.max(torch.min(a_lab, high), low)
        xs.append(student_obs(obs))
        acts.append(a)
        labels.append(a_lab)
        alive.append(~succ_once)
        obs, _, _, _, infos = venv.step(a)
        # the vector env auto-resets on the last step; the real final obs / success flag are in the info dicts
        final = infos["final_observation"] if "final_observation" in infos else obs
        s = (infos["final_info"] if "final_info" in infos else infos)["success"].bool()
        y_obs = final if y_obs is None else torch.where(succ_once[:, None], y_obs, final)  # frozen once done
        if args.feedback == "iou":
            assert isinstance(TASK, PushTTask), "iou feedback is only defined for PushT"
            cur = base.pseudo_render_intersection() if t < T - 1 else (torch.maximum(iou, torch.full_like(iou, 0.9)) * s + iou * ~s)
            iou = cur if iou is None else torch.where(succ_once, iou, cur)
        succ_once |= s
    err = feedback_from_obs(y_obs, "final_error")
    alive = torch.stack(alive, 1)
    xs_t, acts_t, labels_t = torch.stack(xs, 1), torch.stack(acts, 1), torch.stack(labels, 1)
    m = args.n_hints if args.feedback == "hints" else 0
    if args.feedback == "iou":
        y = iou[:, None]
    elif args.feedback in ("final_error_tcp", "init_error_tcp"):
        # PickCubeBias only. [cube - tcp, tcp] at the termination step, so the absolute cube position is a linear function of the token.
        # init_error_tcp uses the cube position at the start of the episode (the hidden task variable) instead of its final position.
        tcp = y_obs[:, 19:22]
        vec = y_obs[:, 36:39] if args.feedback == "final_error_tcp" else (obs0[:, 29:32] - obs0[:, 42:45]) - tcp
        y = torch.cat([vec, tcp], dim=1)
    elif args.feedback == "hints":
        y = torch.zeros(N, 1, device=device)  # constant token: no outcome feedback
    else:
        y = feedback_from_obs(y_obs, args.feedback, succ_once)
    # sparse hints: the m alive steps where the executed action was farthest from the expert's, as (state, expert action)
    disc = ((acts_t - labels_t) ** 2).sum(-1).masked_fill(~alive, -1.0)
    top = disc.topk(m, dim=1).indices if m else torch.zeros(N, 0, dtype=torch.long, device=device)
    hint = torch.cat([xs_t.gather(1, top[..., None].expand(-1, -1, xs_t.shape[-1])),
                      labels_t.gather(1, top[..., None].expand(-1, -1, labels_t.shape[-1]))], dim=-1)
    hint_valid = disc.gather(1, top) >= 0
    return dict(
        obs=xs_t, act=acts_t, lab=labels_t, y=y, alive=alive, hint=hint, hint_valid=hint_valid,
        success_once=succ_once, ep_len=alive.sum(1).float(),
        pos_err=TASK.metrics(err)[0], yaw_err=TASK.metrics(err)[1],
    )


def new_ctx(args, N, S, A, y_dim, device):
    m = args.n_hints if args.feedback == "hints" else 0
    Tc = len(range(args.ctx_stride - 1, args.num_steps, args.ctx_stride)) + (0 if (args.num_steps % args.ctx_stride == 0) else 1)
    return dict(
        steps=torch.zeros(N, args.k, Tc, S + A, device=device), y=torch.zeros(N, args.k, y_dim, device=device),
        valid=torch.zeros(N, args.k, Tc, dtype=torch.bool, device=device),
        hints=torch.zeros(N, args.k, m, S + A, device=device), hint_valid=torch.zeros(N, args.k, m, dtype=torch.bool, device=device),
        n=torch.zeros(N, dtype=torch.long, device=device),
    )


def add_to_ctx(ctx, ep, args):
    i = ctx["n"][0].item()  # all tasks have the same number of episodes
    idx = ctx_step_index(ep["obs"].shape[1], args.ctx_stride, ep["obs"].device)
    ctx["steps"][:, i] = torch.cat([ep["obs"][:, idx], ep["act"][:, idx]], dim=-1)
    ctx["valid"][:, i] = ep["alive"][:, idx]
    ctx["hints"][:, i] = ep["hint"]
    ctx["hint_valid"][:, i] = ep["hint_valid"]
    ctx["y"][:, i] = ep["y"]
    ctx["n"] += 1


def collect_task_episodes(venv, expert, policy, args, goals, seed, student_mode, ctx_mode="normal", frac_prior=0.0):
    """k student episodes with growing context + one expert episode. Returns per-episode logs, context at each stage, expert ep."""
    device = venv.device
    S = student_obs(torch.zeros(1, TASK.obs_dim)).shape[-1]
    A = venv.single_action_space.shape[0]
    y_dim = feedback_dim(args)
    ctx = new_ctx(args, args.num_envs, S, A, y_dim, device)
    eps = []
    for e in range(args.k + (1 if student_mode == "eval" else 0)):
        mode = "student"
        if student_mode == "prior":
            mode = "prior_expert"
        ep = rollout(venv, expert, policy, args, goals, seed + e, mode, ctx, ctx_mode)
        if student_mode == "mixed" and frac_prior > 0:  # replace a random subset of tasks by prior-expert episodes
            pe = rollout(venv, expert, policy, args, goals, seed + e, "prior_expert")
            m = torch.rand(args.num_envs, device=device) < frac_prior
            ep = {key: torch.where(m.view(-1, *[1] * (v.dim() - 1)), pe[key], v) for key, v in ep.items()}
        ep["ctx_n"] = ctx["n"][0].item()
        eps.append(ep)
        if e < args.k:  # in eval mode the last episode is only measured
            add_to_ctx(ctx, ep, args)
    return eps, ctx


def main():
    args = tyro.cli(Args)
    global TASK
    kwargs = {"PickCubeBias-v1": dict(hide_grasp=args.hide_grasp, hide_cube=args.hide_cube), "PegInsertionSideBias-v1": dict(hide_hole=args.hide_hole)}
    TASK = TASKS[args.env_id](**kwargs.get(args.env_id, {}))
    device = torch.device("cuda")
    run = RunDir(args.out_dir or str(LOG_ROOT / "runs" / f"icl-{args.exp_name}"), args)
    torch.manual_seed(args.seed)

    env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="state", control_mode="pd_joint_delta_pos", sim_backend="physx_cuda")
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    base = env.unwrapped
    venv = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True, record_metrics=True)
    obs_dim, act_dim = venv.single_observation_space.shape[0], venv.single_action_space.shape[0]
    assert obs_dim == TASK.obs_dim, obs_dim

    class _ExpertSpace:  # the expert only sees the clean task obs (PickCubeBias appends the hidden offset)
        single_observation_space = type("S", (), {"shape": (TASK.expert_obs(torch.zeros(1, obs_dim)).shape[-1],)})
        single_action_space = venv.single_action_space

    expert = Agent(_ExpertSpace).to(device)
    expert.load_state_dict(torch.load(args.expert_checkpoint, map_location=device))
    expert.eval()
    s_dim, y_dim = student_obs(torch.zeros(1, obs_dim)).shape[-1], feedback_dim(args)
    policy = ICLPolicy(s_dim, act_dim, y_dim, args).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_iterations * args.grad_steps, eta_min=args.lr * 0.02)

    with torch.random.fork_rng():
        torch.manual_seed(args.eval_seed + 1000)  # not an env reset seed: sharing one would correlate goals with the initial states
        eval_goals = TASK.sample(base, args.num_envs)

    def evaluate(ctx_mode="normal"):
        policy.eval()
        eps, _ = collect_task_episodes(venv, expert, policy, args, eval_goals, args.eval_seed, "eval", ctx_mode)
        policy.train()
        return [
            dict(h=i, success_once=ep["success_once"].float().mean().item(), ep_len=ep["ep_len"].mean().item(),
                 pos_err=ep["pos_err"].mean().item(), yaw_err=ep["yaw_err"].mean().item())
            for i, ep in enumerate(eps)
        ]

    # reference levels: expert with the true goal and expert told the default goal (the best a context-free policy could hope for)
    refs = {}
    for name, mode in (("expert_true_goal", "expert"), ("expert_default_goal", "prior_expert")):
        ep = rollout(venv, expert, policy, args, eval_goals, args.eval_seed, mode)
        refs[name] = dict(success_once=ep["success_once"].float().mean().item(), ep_len=ep["ep_len"].mean().item())
        print(f"reference {name:20s} {refs[name]}", flush=True)

    # aggregated dataset, per task: query episodes (k student + 1 expert), their expert labels, and the context available at each
    D = dict(q_obs=[], q_lab=[], q_alive=[], ctx_steps=[], ctx_y=[], ctx_valid=[], ctx_hints=[], ctx_hint_valid=[])
    log = []
    gen_seed = args.seed * 1000
    for it in range(args.num_iterations):
        t0 = time.time()
        goals = TASK.sample(base, args.num_envs)
        seed = gen_seed + it * 10
        eps, ctx = collect_task_episodes(
            venv, expert, policy, args, goals, seed, "prior" if it == 0 else "mixed", frac_prior=args.prior_expert_frac)
        ex = rollout(venv, expert, policy, args, goals, seed + 9, "expert")
        D["q_obs"].append(torch.stack([e["obs"] for e in eps] + [ex["obs"]], 1))  # (N, k+1, T, S)
        D["q_lab"].append(torch.stack([e["lab"] for e in eps] + [ex["lab"]], 1))
        D["q_alive"].append(torch.stack([e["alive"] for e in eps] + [ex["alive"]], 1))  # (N, k+1, T)
        D["ctx_steps"].append(ctx["steps"])
        D["ctx_y"].append(ctx["y"])
        D["ctx_valid"].append(ctx["valid"])
        D["ctx_hints"].append(ctx["hints"])
        D["ctx_hint_valid"].append(ctx["hint_valid"])
        Q, L, QA = torch.cat(D["q_obs"]), torch.cat(D["q_lab"]), torch.cat(D["q_alive"])
        CS, CY, CV = torch.cat(D["ctx_steps"]), torch.cat(D["ctx_y"]), torch.cat(D["ctx_valid"])
        CH, CHV = torch.cat(D["ctx_hints"]), torch.cat(D["ctx_hint_valid"])
        if it == 0:
            policy.set_norm(Q.flatten(0, 1), CY.flatten(0, 1))
        M = Q.shape[0]
        losses = []
        for _ in range(args.grad_steps):
            idx = torch.randint(0, M, (args.batch_size,), device=device)
            j = torch.randint(0, args.k + 1 if args.relabel_student else 1, (args.batch_size,), device=device)
            if not args.relabel_student:
                j = torch.full_like(j, args.k)
            h = torch.where(j == args.k, torch.randint(0, args.k + 1, (args.batch_size,), device=device), j)
            memory, pad = policy.encode_context(CS[idx], CY[idx], h, CV[idx], CH[idx], CHV[idx])
            pred = policy.decode(Q[idx, j], memory, pad)
            m = QA[idx, j].float()  # only steps before the episode terminated
            loss = (((pred - L[idx, j]) ** 2).mean(-1) * m).sum() / m.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        ev = evaluate()
        row = dict(iteration=it, loss=sum(losses[-100:]) / 100, n_tasks=M, eval=ev, seconds=time.time() - t0,
                   train_student_success=[e["success_once"].float().mean().item() for e in eps])
        log.append(row)
        print(f"it {it:2d} loss {row['loss']:.4f} tasks {M} | eval success_once by h: "
              + " ".join(f"{e['success_once']:.3f}" for e in ev) + " | ep_len: " + " ".join(f"{e['ep_len']:.0f}" for e in ev)
              + f" | {row['seconds']:.0f}s", flush=True)
        metrics = dict(loss=row["loss"], n_tasks=M, seconds=row["seconds"])
        for e in ev:
            for key in ("success_once", "ep_len", "pos_err", "yaw_err"):
                metrics[f"eval/{key}_h{e['h']}"] = e[key]
        for i, v in enumerate(row["train_student_success"]):
            metrics[f"train/success_once_ep{i}"] = v
        run.log(it, metrics)
        torch.save(policy.state_dict(), run.ckpt("policy_latest.pt"))

    # controls at the end
    controls = {"normal": log[-1]["eval"]}
    for cm in ("shuf_y", "no_y", "no_ctx"):
        controls[cm] = evaluate(cm)
        print(f"control {cm:7s} success_once by h: " + " ".join(f"{e['success_once']:.3f}" for e in controls[cm])
              + " | ep_len: " + " ".join(f"{e['ep_len']:.0f}" for e in controls[cm]), flush=True)
    torch.save(policy.state_dict(), run.ckpt("policy_final.pt"))
    run.save_json("results.json", dict(refs=refs, controls=controls, final_eval=log[-1]["eval"]))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axs, ("success_once", "ep_len")):
        for cm, ev in controls.items():
            ax.plot([e["h"] for e in ev], [e[key] for e in ev], marker="o", label=cm)
        ax.axhline(refs["expert_true_goal"][key], color="k", ls="--", label="expert (true goal)")
        ax.axhline(refs["expert_default_goal"][key], color="gray", ls=":", label="expert (default goal)")
        ax.set_xlabel("episode index h (context = h previous episodes)")
        ax.set_ylabel(key)
        ax.set_ylim(0, 1 if key == "success_once" else args.num_steps)
    axs[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(run.plot("icl_curve.png"), dpi=120)


if __name__ == "__main__":
    main()
