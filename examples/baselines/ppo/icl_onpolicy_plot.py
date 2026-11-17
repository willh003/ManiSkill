"""Plots for the on-policy ICL loop (icl_onpolicy.py runs logs/runs/icl-op-<kind>-<pairing>-<supervise>-d<demos>-s<seed>).
Writes logs/renders/icl-onpolicy/*.png; every figure exists for pass@k (best of attempts 1..k) and perf@k (success of attempt k alone),
shows the mean over seeds with the seed range as a band, and varies exactly one thing. Missing runs are skipped.
Usage: python icl_onpolicy_plot.py [run-name prefix, default icl-op]"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from run_logging import LOG_ROOT

PFX = sys.argv[1] if len(sys.argv) > 1 else "icl-op"
OUT = f"{LOG_ROOT}/renders/icl-onpolicy"; os.makedirs(OUT, exist_ok=True)
C1, C2, C3, INK, MUTE = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
SEQ = LinearSegmentedColormap.from_list("b", ["#b9d6f5", "#2a78d6", "#0c2f66"])
MET = {"pass_k": "pass@k (success in any of attempts 1..k)", "succ": "perf@k (success of attempt k alone)"}
KN = {"goal": "[f+c]", "fc": "[f, c]", "hist_c": "[h, f, c]"}
DEMOS = [100, 200, 2000]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTE, "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def load(kind="fc", pair="matched", sup="demo", d=100):
    rs = [json.load(open(p)) for p in sorted(glob.glob(f"{LOG_ROOT}/runs/{PFX}-{kind}-{pair}-{sup}-d{d}-s*/results.json"))]
    return [r for r in rs if r.get("rounds")]


def last(rs):
    return min(max(int(k) for k in r["rounds"]) for r in rs)


def arr(rs, rnd, key):
    return np.array([r["rounds"][str(rnd)][key] for r in rs if str(rnd) in r["rounds"] and key in r["rounds"][str(rnd)]])


BCC = "#52514e"
BCTAG = {100: "100", 200: "200", 2000: "2k"}


def bcv(d, m):
    """Visible-hole transformer BC with action noise 0.05 redrawn each retry (icl_fair_bc.py, same 512 eval tasks, 3 seeds): (seeds, K) or None."""
    p = f"{LOG_ROOT}/runs/icl-fair-bc-{BCTAG[d]}.json"
    if not os.path.exists(p):
        return None
    return np.array([v[m] for k, v in json.load(open(p)).items() if k.endswith("-bc_noise0.05")])


def bcline(a, d, m, col=BCC, lab=None):
    v = bcv(d, m)
    if v is not None:
        line(a, v, col, lab or "BC + noise 0.05 (hole visible)", ls="--", lw=1.8, end="BC")


def line(a, v, col, lab, ls="-", lw=2, end=None):
    if not len(v):
        return
    x = np.arange(1, v.shape[1] + 1)
    a.plot(x, v.mean(0), ls, color=col, lw=lw, marker="o", ms=4.5, mec="#fcfcfb", mew=1, label=f"{lab}  (n={len(v)})")
    a.fill_between(x, v.min(0), v.max(0), color=col, alpha=.13, lw=0)
    if end:
        a.annotate(end, (x[-1], v.mean(0)[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)


def fig(title, sub, ylabel, xlabel="k (attempts)"):
    f, a = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    a.set_title(title + "\n", fontsize=11.5, loc="left", color=INK, fontweight="bold")
    a.text(0, 1.025, sub, transform=a.transAxes, fontsize=8.5, color=MUTE)
    a.set_xlabel(xlabel), a.set_ylabel(ylabel), a.set_ylim(0, 1), a.grid(alpha=.22, lw=.6)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    return f, a


def done(f, a, name, loc="upper left"):
    if a.get_legend_handles_labels()[0]:
        if loc == "below":  # crowded plots: legend under the axes
            f.set_size_inches(7.2, 5.6)
            a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
        else:
            a.legend(fontsize=8.5, loc=loc, frameon=False)
        f.savefig(f"{OUT}/{name}.png", dpi=140)
        print("wrote", name)
    plt.close(f)


for m, ml in MET.items():
    # 1. outer loop: one line per round
    for d in DEMOS:
        rs = load(d=d)
        if not rs: continue
        R = last(rs)
        f, a = fig(f"Outer loop: each on-policy round, {d} demos", "[f, c] context, task-matched pairing. Round 1 trains on visible-hole BC failures only.", ml)
        for r in range(1, R + 1):
            line(a, arr(rs, r, m), SEQ((r - 1) / max(R - 1, 1)), f"round {r}", end=f"r{r}" if r in (1, R) else None)
        bcline(a, d, m)
        done(f, a, f"1_outer_loop_d{d}_{m}")
    # 2. rounds summary: metric at k=8 against round
    f, a = fig("Outer loop: k=8 against round", "[f, c] context, task-matched pairing. One line per expert-demo count.", ml.split(" (")[0].replace("@k", "@8"), "round (model trained on data from rounds 0..r-1)")
    for d, col in zip(DEMOS, (C1, C2, C3)):
        rs = load(d=d)
        if not rs: continue
        v = np.stack([arr(rs, r, m)[:, -1] for r in range(1, last(rs) + 1)], 1)
        line(a, v, col, f"{d} demos", end=str(d))
        b = bcv(d, m)
        if b is not None:
            a.plot([1, v.shape[1]], [b[:, -1].mean()] * 2, "--", color=col, lw=1.4, label=f"BC + noise 0.05, {d} demos (k=8)")
    done(f, a, f"2_rounds_summary_{m}", "below")
    # 3. pairing
    for kind in ("fc", "hist_c"):
        for d in DEMOS:
            A, B = load(kind, "matched", d=d), load(kind, "random", d=d)
            if not (A and B): continue
            R = min(last(A), last(B))
            f, a = fig(f"Pairing: task-matched vs random, {d} demos", f"{KN[kind]} context, round {R}. Random = failure from another task, c recomputed for the target demo.", ml)
            line(a, arr(A, R, m), C1, "task-matched", end="matched"), line(a, arr(B, R, m), C2, "random", end="random")
            bcline(a, d, m)
            done(f, a, f"3_pairing_{kind}_d{d}_{m}")
    # 4. demo count
    f, a = fig("Expert demo count", "[f, c] context, task-matched pairing, last round.", ml)
    for d, col in zip(DEMOS, (C1, C2, C3)):
        rs = load(d=d)
        if rs: line(a, arr(rs, last(rs), m), col, f"{d} demos", end=str(d))
        bcline(a, d, m, col, f"BC + noise 0.05, {d} demos")
    done(f, a, f"4_demos_{m}", "below")
    # 5. context type
    for d in DEMOS:
        f, a = fig(f"Context type, {d} demos", "Task-matched pairing, last round. f = failed final peg pose, c = goal - f, h = peg history.", ml)
        for kind, col in zip(("goal", "fc", "hist_c"), (C1, C2, C3)):
            rs = load(kind, d=d)
            if rs: line(a, arr(rs, last(rs), m), col, KN[kind], end=KN[kind])
        bcline(a, d, m)
        done(f, a, f"5_context_d{d}_{m}")
    # 6. sliding window: is the previous attempt used?
    for kind in ("fc", "hist_c"):
        for d in DEMOS:
            rs = load(kind, d=d)
            if not rs: continue
            R = last(rs)
            if not len(arr(rs, R, "randf_" + m)): continue
            f, a = fig(f"Sliding window: is the previous attempt used? {d} demos", f"{KN[kind]} context, task-matched pairing, round {R}. Same model, three contexts at test time.", ml)
            line(a, arr(rs, R, m), C1, "real: previous attempt", end="real")
            line(a, arr(rs, R, "randf_" + m), C2, "another task's failure, exact goal", end="other f")
            line(a, arr(rs, R, "fixed_" + m), C3, "attempt-1 context frozen", end="frozen")
            bcline(a, d, m)
            done(f, a, f"6_window_{kind}_d{d}_{m}")
    # 7. against the fair BC baseline (visible hole, action noise 0.05 per retry)
    for d, tag in zip(DEMOS, ("100", "200", "2k")):
        rs, p = load(d=d), f"{LOG_ROOT}/runs/icl-fair-bc-{tag}.json"
        if not rs or not os.path.exists(p): continue
        fb = json.load(open(p))
        f, a = fig(f"On-policy ICL vs BC with action noise, {d} demos", "ICL: hole hidden, [f, c], task-matched, last round. BC: hole visible, noise 0.05 redrawn each retry.", ml)
        line(a, arr(rs, last(rs), m), C1, "on-policy ICL (hole hidden)", end="ICL")
        bcline(a, d, m, C2)
        done(f, a, f"7_vs_bc_d{d}_{m}", "lower right")
    # 8. dagger relabel vs demo pairing
    for d in DEMOS:
        A, B = load(d=d), load(sup="dagger", d=d)
        if not (A and B): continue
        R = min(last(A), last(B))
        f, a = fig(f"Supervision: expert demos vs DAgger relabel, {d} demos", f"[f, c] context, task-matched pairing, round {R}. Relabel adds expert actions on the student's own states.", ml)
        line(a, arr(A, R, m), C1, "expert demos only", end="demos"), line(a, arr(B, R, m), C2, "+ DAgger relabel", end="relabel")
        bcline(a, d, m)
        done(f, a, f"8_dagger_d{d}_{m}")

# 9. train / val loss per round (seed 1)
for d in DEMOS:
    p = f"{LOG_ROOT}/runs/{PFX}-fc-matched-demo-d{d}-s1/curves.json"
    if not os.path.exists(p): continue
    cv = {k: np.array(v) for k, v in json.load(open(p)).items() if k != "bc" and len(v)}
    f, a = fig(f"Train and validation loss per round, {d} demos", "[f, c], task-matched, seed 1. Solid = held-out demos with held-out on-policy contexts; dashed = train batch.", "action MSE", "gradient step")
    for i, (k, v) in enumerate(sorted(cv.items())):
        col = SEQ(i / max(len(cv) - 1, 1))
        a.plot(v[:, 0], v[:, 2], "-", color=col, lw=2, label=f"val, round {k[1:]}"), a.plot(v[:, 0], v[:, 1], "--", color=col, lw=1.2)
    a.set_yscale("log"), a.set_ylim(1e-4, 1.5)
    done(f, a, f"9_loss_d{d}", "upper right")
