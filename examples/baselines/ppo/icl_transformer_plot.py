"""Aggregate icl_transformer.py runs (logs/runs/icl-tf-<kind>-s<seed>/results.json) over seeds -> table (markdown) + plot.
Usage: python icl_transformer_plot.py [out_dir]   (default logs/renders/icl-tf)"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_logging import LOG_ROOT

OUT = sys.argv[1] if len(sys.argv) > 1 else str(LOG_ROOT / "renders/icl-tf")
os.makedirs(OUT, exist_ok=True)
NS = [100, 1000, 10000]
KINDS = [("fcfc", "T1 [f,c,f+c]", "#2a78d6"), ("fcfc_hist", "A  +history", "#eb6834"), ("hist_c", "B  history + c (no f+c)", "#1baf7a"), ("hist_c_noisy", "C  B + noisy c", "#e87ba4")]
R = {}
for k, _, _ in KINDS:
    for p in sorted(glob.glob(f"{LOG_ROOT}/runs/icl-tf-{k}-s*/results.json")):
        R.setdefault(k, []).append(json.load(open(p)))
BC = [json.load(open(p)) for p in sorted(glob.glob(f"{LOG_ROOT}/runs/icl-tf-fcfc-s*/results.json"))]
tf_bc = [r["tf_bc"]["success"] for r in BC if "tf_bc" in r]
mlp_bc = [r["mlp_bc"] for r in BC if "mlp_bc" in r]

# MLP [f,c,f+c] reference (icl_final_state.py pose_sum, expert norm, 20k steps): N=100/1000 in icl-norm-pose_sum-s*, N=10000 in icl-norm-pose_sum-N10000-s*
mlp = {n: [] for n in NS}
for p in sorted(glob.glob(f"{LOG_ROOT}/runs/icl-norm-pose_sum*-s*/results.json")):
    for n in NS:
        v = json.load(open(p))["runs"].get(f"N{n}")
        if v:
            mlp[n].append(v["pass2"])


def get(k, n, field, i=None):
    vals = []
    for r in R.get(k, []):
        v = r["runs"].get(f"{k}-N{n}")
        if v:
            vals.append(v[field] if i is None else v[field][i])
    return np.array(vals)


def kvec(k, n, field):
    return np.array([r["runs"][f"{k}-N{n}"][field] for r in R.get(k, []) if f"{k}-N{n}" in r["runs"]])


fig, ax = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
# (a) pass@2 vs N
a = ax[0, 0]
for k, name, col in KINDS:
    xs = [n for n in NS if len(get(k, n, "pass_k", 1))]
    m = [get(k, n, "pass_k", 1).mean() for n in xs]
    a.plot(xs, m, "-o", color=col, label=name, lw=2)
    for n in xs:
        a.scatter([n] * len(get(k, n, "pass_k", 1)), get(k, n, "pass_k", 1), color=col, s=12, alpha=.45, zorder=1)
xm = [n for n in NS if mlp[n]]
a.plot(xm, [np.mean(mlp[n]) for n in xm], "--s", color="#52514e", label="MLP [f,c,f+c] (20k)", lw=1.5)
if tf_bc:
    a.axhline(np.mean(tf_bc), color="#2a78d6", ls=":", lw=1), a.text(100, np.mean(tf_bc) + .008, "TF BC (visible goal)", fontsize=8, color="#52514e")
if mlp_bc:
    a.axhline(np.mean(mlp_bc), color="#52514e", ls=":", lw=1), a.text(100, np.mean(mlp_bc) + .008, "MLP BC (rollout source)", fontsize=8, color="#52514e")
a.set_xscale("log"), a.set_xlabel("N suboptimal BC rollouts"), a.set_ylabel("pass@2"), a.set_title("pass@2 vs N (dots = seeds)"), a.legend(fontsize=8, loc="lower right"), a.set_ylim(0, 0.9)
# (b) pass@k at N=1000, (c) best-so-far distance at N=1000
for j, (field, yl, ttl) in enumerate((("pass_k", "pass@k", "pass@k, N=1000"), ("dist_best", "best-so-far peg-goal distance (m)", "distance@k, N=1000"))):
    a = ax[0, 1] if j == 0 else ax[1, 0]
    for k, name, col in KINDS:
        v = kvec(k, 1000, field)
        if len(v):
            a.plot(range(1, 9), v.mean(0), "-o", color=col, label=name, lw=2, ms=4)
            a.fill_between(range(1, 9), v.min(0), v.max(0), color=col, alpha=.15)
    a.set_xlabel("k (attempts)"), a.set_ylabel(yl), a.set_title(ttl + " (band = seed range)")
    if j == 1:
        a.set_yscale("log"), a.set_ylim(top=0.5)
# (d) C only: distance@k per N
a = ax[1, 1]
for n, ls in zip(NS, ("-", "--", ":")):
    v = kvec("hist_c_noisy", n, "dist_best")
    if len(v):
        a.plot(range(1, 9), v.mean(0), ls, color="#e87ba4", lw=2, marker="o", ms=4, label=f"N={n}")
    v = kvec("hist_c_noisy", n, "dist")
    if len(v):
        a.plot(range(1, 9), v.mean(0), ls, color="#52514e", lw=1.2, label=f"attempt-k dist, N={n}")
a.set_yscale("log"), a.set_xlabel("k"), a.set_ylabel("peg-goal distance (m)"), a.set_title("C: best-so-far (pink) and per-attempt (gray) distance"), a.legend(fontsize=7)
for x in ax.ravel():
    x.grid(alpha=.25)
fig.savefig(f"{OUT}/summary.png", dpi=130)

# markdown tables
L = ["| kind | N | seeds | pass@1 | pass@2 | pass@4 | pass@8 | dist_best@2 | dist_best@8 | angle_best@8 | no_ctx p@2 | shuf p@2 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for k, name, _ in KINDS:
    for n in NS:
        pk, db, ab = kvec(k, n, "pass_k"), kvec(k, n, "dist_best"), kvec(k, n, "ang_best")
        if not len(pk):
            continue
        f = lambda v, i: f"{v[:, i].mean():.3f}±{v[:, i].std():.3f}"
        L.append(f"| {name} | {n} | {len(pk)} | {f(pk, 0)} | {f(pk, 1)} | {f(pk, 3)} | {f(pk, 7)} | {f(db, 1)} | {f(db, 7)} | {f(ab, 7)} | {np.mean([r['runs'][f'{k}-N{n}']['no_ctx_pass2'] for r in R[k]]):.3f} | {np.mean([r['runs'][f'{k}-N{n}']['shuf_pass2'] for r in R[k]]):.3f} |")
L.append("")
L.append(f"TF BC (visible goal) success per seed: {[round(x, 3) for x in tf_bc]}; MLP BC (rollout source): {[round(x, 3) for x in mlp_bc]}")
L.append("MLP [f,c,f+c] pass@2 (20k): " + ", ".join(f"N={n}: {np.mean(mlp[n]):.3f} (n={len(mlp[n])})" for n in NS if mlp[n]))
open(f"{OUT}/summary.md", "w").write("\n".join(L))
print("\n".join(L))
