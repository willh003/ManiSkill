"""Standard DAgger control for the recurrent-PPO BC stage (rl2_peg.py bc()).

Student: feedforward MLP on the CLEAN 43-dim expert obs (hole pose included), single episode of 100 steps, no feedback / memory.
Expert: the same peg-expert-r1 (deterministic), label = expert(obs of the visited state). Tasks: the M fixed train holes
(--fixed-tasks M --shared-peg, same pool as ppo_tasks.py / rl2_peg.py). Query budget matched to rl2_peg.py bc(): 12 rounds x 1024 envs x
800 steps = 9,830,400 expert labels (here 8 rollouts of 1024 envs x 100 steps per round). beta (expert share) 1 -> 0 over the first 4 rounds,
1000 Adam steps per round on the aggregated data. Eval each round: deterministic student on the train pool and on the 512 held-out holes
(Generator seed 777), success = first success within 100 steps (same metric as rl2 eval ep1)."""
import json
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import tyro

from icl_tasks import checker_sample
from meta_peg import EP_LEN, MetaPeg, load_expert
from rl2_peg import Norm, mlp
from run_logging import LOG_ROOT, RunDir


@dataclass
class Args:
    expert_checkpoint: str = str(LOG_ROOT / "runs/peg-expert-r1/ckpt_3576.pt")
    out_dir: str = str(LOG_ROOT / "runs/dagger-peg")
    seed: int = 1
    num_envs: int = 1024
    fixed_tasks: int = 4
    pool_seed: int = 0
    rounds: int = 12
    rollouts_per_round: int = 8  # x num_envs x 100 steps = 819,200 labels per round (= rl2_peg.py bc(): 1024 envs x 800 steps)
    beta_rounds: int = 4
    steps: int = 1000
    batch: int = 16384  # transitions per gradient step
    lr: float = 1e-3
    hidden: int = 256
    eval_only: bool = False  # just evaluate the expert checkpoint (train pool / held-out holes, shared peg) and exit


class Student(nn.Module):
    def __init__(self, D, A, H):
        super().__init__()
        self.norm, self.net = Norm(D), mlp(D, A, H, 3, out_std=0.01)

    def forward(self, x):
        return self.net(self.norm(x))


@torch.no_grad()
def rollout(env, expert, student, beta, seed, gen, goals=None, held=False, record=False, policy="student"):
    env.reset(seed, goals=goals, gen=gen, held=held)
    N = env.N
    use_or = torch.rand(N, device=env.dev, generator=gen) < beta
    X, Y = [], []
    for _ in range(EP_LEN):
        x = env.expert_obs()
        lab = expert.get_action(x, deterministic=True)
        if policy == "expert":
            act = lab
        else:
            act = torch.where(use_or[:, None], lab, student(x)) if beta > 0 else student(x)
        if record:
            X.append(x.half()), Y.append(lab.half())
        env.step(act)
    out = dict(succ=env.succ[:, 0].float())
    if record:
        out["x"], out["y"] = torch.cat(X), torch.cat(Y)  # (T*N, D)
    return out


def main(a):
    rd = RunDir(a.out_dir, a)
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    env = MetaPeg(a.num_envs, 1, 0.0, 0.0, "wide_corner", True, fixed_tasks=a.fixed_tasks, pool_seed=a.pool_seed, shared_peg=True)
    expert = load_expert(a.expert_checkpoint, dev)
    D, A = 43, env.A
    stu = Student(D, A, a.hidden).to(dev)
    gen = torch.Generator(device=dev).manual_seed(1000 + a.seed)
    ge = torch.Generator(device=dev).manual_seed(12345)
    hg = checker_sample(512, "all", dev, torch.Generator(device=dev).manual_seed(777), "wide_corner").repeat(math.ceil(a.num_envs / 512), 1)[: a.num_envs]
    tg = env.fixed_pool.to(dev)[torch.arange(512, device=dev) % len(env.fixed_pool)].repeat(math.ceil(a.num_envs / 512), 1)[: a.num_envs]
    seed_ctr = 10 ** 6 * a.seed

    def evaluate(policy="student"):
        r_tr = rollout(env, expert, stu, 0.0, 12345, ge, goals=tg, held=True, policy=policy)["succ"][:512].mean().item()  # exact pool rows, as rl2_peg.py eval()
        r_he = rollout(env, expert, stu, 0.0, 12345, ge, goals=hg, held=True, policy=policy)["succ"][:512].mean().item()
        return r_tr, r_he

    ref_tr, ref_he = evaluate("expert")
    print(f"expert ({a.expert_checkpoint}) train {ref_tr:.3f} held-out {ref_he:.3f}", flush=True)
    if a.eval_only:
        rd.save_json("results.json", dict(expert=dict(train=ref_tr, heldout=ref_he)))
        print("ALLDONE", flush=True)
        return
    opt = torch.optim.Adam(stu.parameters(), lr=a.lr)
    Xs, Ys, nq = [], [], 0
    for it in range(a.rounds):
        beta = max(0.0, 1 - it / max(a.beta_rounds, 1))
        t0 = time.time()
        for k in range(a.rollouts_per_round):
            seed_ctr += 1
            r = rollout(env, expert, stu, beta, seed_ctr, gen, record=True)
            if it == 0 and k == 0:
                stu.norm.set(r["x"].float())
            Xs.append(r["x"]), Ys.append(r["y"])
            nq += len(r["x"])
        X, Y = torch.cat(Xs), torch.cat(Ys)
        M, losses = len(X), []
        for _ in range(a.steps):
            idx = torch.randint(0, M, (a.batch,), device=dev)
            loss = ((stu(X[idx].float()) - Y[idx].float()) ** 2).mean()
            opt.zero_grad(), loss.backward(), nn.utils.clip_grad_norm_(stu.parameters(), 1.0), opt.step()
            losses.append(loss.item())
        tr, he = evaluate()
        m = dict(beta=beta, loss=sum(losses[-50:]) / 50, expert_queries=nq, eval_succ=tr, eval_held_succ=he, rollout_succ=r["succ"].mean().item(), time=time.time() - t0)
        rd.log(it, m)
        print(f"[dagger {it}] " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
    torch.save(stu.state_dict(), rd.ckpt("student.pt"))
    rd.save_json("results.json", dict(final=dict(train=tr, heldout=he), expert=dict(train=ref_tr, heldout=ref_he), expert_queries=nq))
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
