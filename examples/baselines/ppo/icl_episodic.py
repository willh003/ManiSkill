"""Episodic in-context learning (a hidden task variable, corrective feedback after each episode).

A "task" is a hidden variable the student cannot observe (the expert can); task definitions are in icl_tasks.py (PushTGoal-v1: goal T pose,
PickCubeBias-v1, PegInsertionSideBias-v1, RotateDirHidden-v1). The student rolls out up to k episodes on the same task;
after each episode i it receives feedback y_i: the privileged final-state error (task-specific), the success flag, or expert-action hints. Context c_h = ((xi_0, y_0), ..., (xi_{h-1}, y_{h-1})). The student is a transformer
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
    python icl_episodic.py --exp-name icl --env-id PickCubeBias-v1 --expert-checkpoint <ppo ckpt> --num-steps 50
"""
import re
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

from icl_tasks import LANG_WORDS, TASKS, PushTTask, yaw_of_quat
from ppo import Agent
from run_logging import LOG_ROOT, RunDir

TASK = PushTTask()
CTX_POLS = []  # policies that roll out the context episodes (--ctx-policies), in the order listed


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
    hide_hole: bool = False
    """PegInsertionSideBias only: the student gets no hole pose, and the task is the box pose (x, y, yaw) (else: biased observed hole pose)"""
    full_info: bool = False
    """PegInsertionSideBias only: nothing is hidden (the student sees the true hole pose; the box pose is fixed per task, random across tasks)"""
    ctx_policies: str = ""
    """comma- or colon-separated policy checkpoints (or 'expert' = --expert-checkpoint), one per scripted episode (at most k): the first m episodes of a
    task are rolled out (deterministically) by these instead of by the student; the remaining episodes are the student's own"""
    ctx_random_order: bool = False
    """with --ctx-policies: use an independent random permutation of the listed policies for every task (default: the listed order)"""
    feedback: str = "final_error"
    """final_error: the task's privileged final-state error (see icl_tasks.py); success: the 0/1 success flag; hints: no outcome feedback, instead n_hints (state, expert action) pairs per episode
    at the steps where the student's action differed most from the expert's; language (PegTask): the final error as a templated text command, e.g.
    "move the peg 4.2 cm forward and 1.5 cm right and 0.3 cm down then turn it 12 degrees clockwise", fed to the student as word tokens"""
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


def feedback_dim(args):
    return {"final_error": lambda: TASK.y_dim, "success": lambda: 1, "hints": lambda: 1, "language": lambda: TASK.lang_len}[args.feedback]()


def feedback_from_obs(obs, kind, success=None):
    """Feedback computed from the (privileged) final observation, which still contains the goal and the T pose."""
    if kind == "success":
        return success.float()[:, None]
    if kind == "language":
        return TASK.language(obs).float()  # word ids (0 = pad), stored in the float feedback buffers
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
        self.lang = args.feedback == "language"
        self.y_in = nn.Linear(y_dim, d)
        self.word_emb = nn.Embedding(len(LANG_WORDS), d)
        self.word_pos = nn.Embedding(y_dim, d)  # y_dim = number of words of a command
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
        Returns memory (B, 1 + k*(Tc+M+m), d), M feedback tokens per episode (1, or the number of words) and a key padding mask (True = ignore)."""
        B, k, Tc, _ = ctx_steps.shape
        dev = ctx_steps.device
        s = (ctx_steps[..., : self.s_mean.shape[0]] - self.s_mean) / self.s_std
        x = torch.cat([s, ctx_steps[..., self.s_mean.shape[0] :]], dim=-1)
        ep = self.ep_emb(torch.arange(k, device=dev))[None, :, None]  # (1,k,1,d)
        pos = self.pos_emb(torch.arange(Tc, device=dev))[None, None]  # (1,1,Tc,d)
        step_tok = self.step_in(x) + ep + pos + self.type_emb.weight[1]
        if self.lang:  # (B,k,L) word ids -> L word tokens per episode; pad words are masked
            ids = ctx_y.long()
            y_tok = self.word_emb(ids) + self.word_pos(torch.arange(ids.shape[-1], device=dev))[None, None] + ep + self.type_emb.weight[2]
            y_valid = ids > 0
        else:
            y_tok = self.y_in((ctx_y - self.y_mean) / self.y_std)[:, :, None] + ep + self.type_emb.weight[2]
            y_valid = torch.ones(B, k, 1, dtype=torch.bool, device=dev)
        h = torch.cat([(hints[..., : self.s_mean.shape[0]] - self.s_mean) / self.s_std, hints[..., self.s_mean.shape[0] :]], dim=-1)
        hint_tok = self.hint_in(h) + ep + self.type_emb.weight[3]
        m = hints.shape[2]
        tokens = torch.cat([step_tok, y_tok, hint_tok], dim=2).reshape(B, k * (Tc + y_tok.shape[2] + m), -1)
        ep_valid = (torch.arange(k, device=dev)[None] < n_eps[:, None])[:, :, None]  # (B,k,1)
        valid = torch.cat([ctx_valid & ep_valid, y_valid & ep_valid, hint_valid & ep_valid], dim=2).reshape(B, -1)
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


def rollout(venv, expert, policy, args, goals, seed, mode, ctx=None, ctx_mode="normal", pol_idx=None):
    """One episode on every task. mode: 'student' | 'expert' | 'prior_expert' (expert told the default goal) | 'scripted' (CTX_POLS[pol_idx[env]]).
    ctx: dict(steps (N,k,Tc,S+A), y (N,k,Y), n (N,)) of previous episodes; ctx_mode: normal | shuf_y | no_y | no_ctx.
    Returns states, executed actions, expert labels, success flags and the feedback for this episode."""
    device, N, T = venv.device, args.num_envs, args.num_steps
    low = torch.from_numpy(venv.single_action_space.low).to(device)
    high = torch.from_numpy(venv.single_action_space.high).to(device)
    obs, _ = venv.reset(seed=seed, options=TASK.reset_options(goals))
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
    base = venv.unwrapped
    for t in range(T):
        with torch.no_grad():
            a_lab = expert.get_action(TASK.expert_obs(obs), deterministic=True)
            if mode == "expert":
                a = a_lab
            elif mode == "prior_expert":
                a = expert.get_action(TASK.expert_obs(TASK.prior_obs(obs, base)), deterministic=True)
            elif mode == "scripted":
                a = a_lab
                for i, pol in enumerate(CTX_POLS):
                    a = torch.where((pol_idx == i)[:, None], pol.get_action(TASK.expert_obs(obs), deterministic=True), a)
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
        succ_once |= s
    err = feedback_from_obs(y_obs, "final_error")
    alive = torch.stack(alive, 1)
    xs_t, acts_t, labels_t = torch.stack(xs, 1), torch.stack(acts, 1), torch.stack(labels, 1)
    m = args.n_hints if args.feedback == "hints" else 0
    if args.feedback == "hints":
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


def ctx_order(args, N, device, m, gen=None):
    """(N, m) index into CTX_POLS of the policy that rolls out each scripted context episode of each task."""
    if args.ctx_random_order:
        return torch.rand(N, m, device=device, generator=gen).argsort(1)
    return torch.arange(m, device=device)[None].expand(N, -1)


def collect_scripted_episodes(venv, expert, policy, args, goals, seed, student_mode, ctx_mode="normal", frac_prior=0.0, gen=None):
    """The first m = len(CTX_POLS) episodes of a task are rolled out by CTX_POLS, the others (up to k) by the student. The student's
    query episode h sees the first h episodes of the context; for h < m that is a separate rollout (the scripted episode is the context),
    for h >= m it is the episode that is then appended to the context. Evaluation adds the query h = k. Returns the student episodes
    (each with its own context size h) and the full context."""
    device, N, m = venv.device, args.num_envs, len(CTX_POLS)
    S = student_obs(torch.zeros(1, TASK.obs_dim)).shape[-1]
    A = venv.single_action_space.shape[0]
    ctx = new_ctx(args, N, S, A, feedback_dim(args), device)
    order = ctx_order(args, N, device, m, gen)

    def student_episode(h):
        c = dict(ctx, n=torch.full_like(ctx["n"], h))
        mode = "prior_expert" if student_mode == "prior" else "student"
        ep = rollout(venv, expert, policy, args, goals, seed + 4 + h, mode, c, ctx_mode)
        if student_mode == "mixed" and frac_prior > 0:
            pe = rollout(venv, expert, policy, args, goals, seed + 4 + h, "prior_expert")
            mask = torch.rand(N, device=device) < frac_prior
            ep = {key: torch.where(mask.view(-1, *[1] * (v.dim() - 1)), pe[key], v) for key, v in ep.items()}
        ep["ctx_n"] = h
        return ep

    eps = []
    for e in range(args.k):
        if e < m:
            add_to_ctx(ctx, rollout(venv, expert, policy, args, goals, seed + e, "scripted", pol_idx=order[:, e]), args)
        else:
            eps.append(student_episode(e))
            add_to_ctx(ctx, eps[-1], args)
    eps = [student_episode(h) for h in range(min(m, args.k))] + eps  # queries at h < m, after the scripted context exists
    if student_mode == "eval":
        eps.append(student_episode(args.k))
    return eps, ctx


def main():
    args = tyro.cli(Args)
    global TASK
    kwargs = {"PegInsertionSideBias-v1": dict(hide_hole=args.hide_hole, full_info=args.full_info)}
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
    if args.ctx_policies:
        names = re.split("[,:]", args.ctx_policies)
        assert 1 <= len(names) <= args.k, "--ctx-policies: one entry per scripted context episode (the first ones), at most k"
        for n in names:
            if n == "expert":
                CTX_POLS.append(expert)
                continue
            pol = Agent(_ExpertSpace).to(device)
            pol.load_state_dict(torch.load(n, map_location=device))
            pol.eval()
            CTX_POLS.append(pol)
    collect = collect_scripted_episodes if args.ctx_policies else collect_task_episodes
    s_dim, y_dim = student_obs(torch.zeros(1, obs_dim)).shape[-1], feedback_dim(args)
    policy = ICLPolicy(s_dim, act_dim, y_dim, args).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_iterations * args.grad_steps, eta_min=args.lr * 0.02)

    with torch.random.fork_rng():
        torch.manual_seed(args.eval_seed + 1000)  # not an env reset seed: sharing one would correlate goals with the initial states
        eval_goals = TASK.sample(base, args.num_envs)

    def evaluate(ctx_mode="normal"):
        policy.eval()
        extra = {}
        if args.ctx_policies:  # same task orders for every evaluation
            extra["gen"] = torch.Generator(device=device).manual_seed(args.eval_seed)
        eps, _ = collect(venv, expert, policy, args, eval_goals, args.eval_seed, "eval", ctx_mode, **extra)
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

    if args.ctx_policies:  # how good are the context policies on the eval tasks (success_once)
        for i, name in enumerate(re.split("[,:]", args.ctx_policies)):
            ep = rollout(venv, expert, policy, args, eval_goals, args.eval_seed, "scripted", pol_idx=torch.full((args.num_envs,), i, device=device))
            refs[f"ctx_policy_{i}_{name.split('/')[-1]}"] = dict(success_once=ep["success_once"].float().mean().item(), ep_len=ep["ep_len"].mean().item())
            print(f"reference ctx policy {i} {name:40s} {refs[f'ctx_policy_{i}_{name.split(chr(47))[-1]}']}", flush=True)

    # aggregated dataset, per task: query episodes (k student + 1 expert), their expert labels, and the context available at each
    D = dict(q_obs=[], q_lab=[], q_alive=[], ctx_steps=[], ctx_y=[], ctx_valid=[], ctx_hints=[], ctx_hint_valid=[])
    log = []
    gen_seed = args.seed * 1000
    for it in range(args.num_iterations):
        t0 = time.time()
        goals = TASK.sample(base, args.num_envs)
        seed = gen_seed + it * 10
        eps, ctx = collect(
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
