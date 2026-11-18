"""Memory analysis of a trained rl2_peg policy. Usage: python rl2_memory.py RUN_DIR OUT.json
1. Interventions on the GRU state / observed feedback (success per episode, 512 eval tasks, deterministic).
2. Linear (ridge) probes of the GRU state at episode starts and within episodes vs the same probe on the observation."""
import json, sys, copy
import numpy as np, torch
from rl2_peg import Args, Trainer
from meta_peg import S_DIM, EP_LEN

run, out = sys.argv[1], sys.argv[2]
a = json.load(open(f"{run}/config.json"))["args"]
a["num_envs"] = 1024
tr = Trainer(Args(**a))
ck = torch.load(f"{run}/checkpoints/latest.pt", map_location=tr.dev)
tr.actor.load_state_dict(ck["actor"])
env, n_ep, dev = tr.env, tr.env.n_ep, tr.dev
FC = slice(S_DIM + n_ep + 1, S_DIM + n_ep + 1 + 14)


def shift(N, k=137):
    return torch.roll(torch.arange(N, device=dev), k)


@torch.no_grad()
def rollout(goals, seed, hreset=(), hswap=(), fb=None, K=None, rec_t=None):
    """hreset/hswap: episode indices at whose start h is zeroed / permuted across tasks. fb: {ep: 'zero'|'shuf'|'noh'} obs feedback ablation.
    K: reset h every K steps. rec_t: steps (global, within-ep t) at which to record (h, obs, targets)."""
    a_obs, _ = env.reset(seed, goals, tr.gen)
    N = env.N
    h, perm = None, shift(N)
    R = dict(h=[], x=[], ep=[], t=[])
    for t in range(env.T):
        e, tt = env.ep, env.t
        if tt == 0:
            if e in hreset:
                h = None
            if e in hswap and h is not None:
                h = h[:, perm]
        if K and tt % K == 0:
            h = None
        x = a_obs
        if fb and e in fb:
            x = x.clone()
            if fb[e] == "shuf":
                x[:, FC] = x[perm, FC]
            else:
                x[:, FC] = 0
                x[:, -1] = 0
        if rec_t is not None and tt in rec_t:
            R["h"].append((h[0] if h is not None else torch.zeros(N, tr.args.hidden, device=dev)).cpu()), R["x"].append(x.cpu())
            R["ep"].append(e), R["t"].append(tt)
        mean, h = tr.actor(x[:, None], h)
        a_obs, _, _, _, _ = env.step(mean[:, 0])
    R["succ"] = env.succ.float().cpu()
    R["f"], R["c"] = [f.cpu() for f in env.f_hist], [c.cpu() for c in env.c_hist]
    R["goals"] = goals.cpu()
    return R


def evalcfg(**kw):
    R = rollout(tr.eval_goals, a["eval_seed"], **kw)
    s = R["succ"][:512]
    return dict(succ=[round(v, 3) for v in s.mean(0).tolist()], pass_n=round(s.max(1).values.mean().item(), 3))


import os
PO = os.environ.get("PROBES_ONLY")
res = dict(run=run, n_ep=n_ep, noise=a["fb_noise_xy"], interventions={})
I = res["interventions"]
if not PO:
  I["base"] = evalcfg()
  I["hreset_all_boundaries"] = evalcfg(hreset=set(range(1, n_ep)))
  for e in range(1, n_ep):
      I[f"hreset_start_ep{e + 1}"] = evalcfg(hreset={e})
      I[f"hswap_start_ep{e + 1}"] = evalcfg(hswap={e})   # other task's memory, own observed feedback
      I[f"obs_fb_zero_ep{e + 1}"] = evalcfg(fb={e: "zero"})   # keep memory, hide observed (f, c)
      I[f"obs_fb_shuf_ep{e + 1}"] = evalcfg(fb={e: "shuf"})  # other task's observed (f, c), own memory
      I[f"hreset+obs_fb_zero_ep{e + 1}"] = evalcfg(hreset={e}, fb={e: "zero"})
  for K in (1, 2, 5, 10, 25, 50):
      I[f"hreset_every_{K}_steps"] = evalcfg(K=K)
  for k, v in I.items():
      print(k, v, flush=True)




# ---- probes
def ridge_r2(X, Y, seed=0):
    X, Y = X.numpy().astype(np.float64), Y.numpy().astype(np.float64)
    n = len(X)
    idx = np.random.RandomState(seed).permutation(n)
    tr_i, te_i = idx[: int(.75 * n)], idx[int(.75 * n):]
    mu, sd = X[tr_i].mean(0), X[tr_i].std(0) + 1e-6
    Xn = (X - mu) / sd
    best = -9
    for lam in (1., 10., 100., 1000.):  # choose on a val slice of train
        v = tr_i[: len(tr_i) // 5]; t2 = tr_i[len(tr_i) // 5:]
        W = np.linalg.solve(Xn[t2].T @ Xn[t2] + lam * np.eye(X.shape[1]), Xn[t2].T @ (Y[t2] - Y[t2].mean(0)))
        r = 1 - ((Xn[v] @ W + Y[t2].mean(0) - Y[v]) ** 2).sum() / ((Y[v] - Y[v].mean(0)) ** 2).sum()
        if r > best: best, bl = r, lam
    W = np.linalg.solve(Xn[tr_i].T @ Xn[tr_i] + bl * np.eye(X.shape[1]), Xn[tr_i].T @ (Y[tr_i] - Y[tr_i].mean(0)))
    P = Xn[te_i] @ W + Y[tr_i].mean(0)
    return float(1 - ((P - Y[te_i]) ** 2).sum() / ((Y[te_i] - Y[te_i].mean(0)) ** 2).sum())


REC = [0, 10, 30, 60, 99]
Rs = []
for i in range(3):
    g = torch.stack(__import__("icl_tasks").checker_sample(1024, "all", dev, tr.gen, a["split"]).unbind(0)) if False else \
        __import__("icl_tasks").checker_sample(1024, "all", dev, tr.gen, a["split"])
    Rs.append(rollout(g, 777 + i, rec_t=REC))
cat = lambda k: torch.cat([r[k] for r in Rs])
H = [torch.cat([r["h"][j] for r in Rs]) for j in range(len(Rs[0]["h"]))]
X = [torch.cat([r["x"][j] for r in Rs]) for j in range(len(Rs[0]["x"]))]
goals = cat("goals")
F = [cat_ for cat_ in [torch.cat([r["f"][k] for r in Rs]) for k in range(n_ep)]]
C = [torch.cat([r["c"][k] for r in Rs]) for k in range(n_ep)]
est = [F[k] + C[k] for k in range(n_ep)]  # the (noisy) hole estimate each feedback implies
tg = {"hole_xy": goals[:, :2], "hole_yaw": goals[:, 2:3]}
for k in range(n_ep - 1):
    tg[f"est{k + 1}_xy (f+c of ep{k + 1})"] = est[k][:, :2]
    tg[f"c{k + 1}_xy"] = C[k][:, :2]
    tg[f"f{k + 1}_xy"] = F[k][:, :2]
tg["noise_hole_est1 (est1-hole, xy)"] = est[0][:, :2] - goals[:, :2]
if n_ep > 2:
    tg["est1-est2_xy"] = est[0][:, :2] - est[1][:, :2]
    tg["mean_est_xy"] = torch.stack(est[: n_ep - 1]).mean(0)[:, :2]
P = {}
meta = list(zip(Rs[0]["ep"], Rs[0]["t"]))
for j, (e, t) in enumerate(meta):
    if e == 0 and t != 99 and t != 0:
        continue
    row = {}
    for name, Y in tg.items():
        row[name] = dict(h=round(ridge_r2(H[j], Y), 3), obs=round(ridge_r2(X[j], Y), 3), both=round(ridge_r2(torch.cat([H[j], X[j]], 1), Y), 3))
    P[f"ep{e + 1}_t{t}"] = row
    print(f"ep{e + 1} t{t}", {k: v for k, v in row.items()}, flush=True)
res["probes"] = P
json.dump(res, open(out, "w"), indent=1)
