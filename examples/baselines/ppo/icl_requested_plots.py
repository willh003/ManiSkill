"""3 requested plots -> logs/renders/icl-tf/{plot1_t1_vs_bc,plot2_pass2_vs_N,plot3_variants}.png"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_logging import LOG_ROOT

OUT = f"{LOG_ROOT}/renders/icl-tf"; os.makedirs(OUT, exist_ok=True)
MODE = sys.argv[1] if len(sys.argv) > 1 else "tok"  # tok | cat (concat, 500 demos) | cat2k (concat, 2000 demos, MLP also 2000 demos)
SUF, DIR = {"tok": ("", None), "cat": ("_cat", "icl-tf-dbg-catabc"), "cat2k": ("_cat2k", "icl-tf-dbg-catabc2k"), "cat2kb": ("_cat2kb", "icl-tf-dbg-catabc2k"), "cat100": ("_cat100", "icl-tf-dbg-c100-{k}")}[MODE]
BIAS100 = MODE == "cat100"
BIAS = MODE in ("cat2kb", "cat100")  # add C with biased scale U(1.1,1.3) (icl-tf-dbg-cbias2k)
MODE = "cat2k" if MODE == "cat2kb" else MODE
TTL = {"tok": " [token pathway]", "cat": " [concat pathway]", "cat2k": " [concat, 2000 demos]", "cat100": " [concat, 100 demos]"}[MODE]
BCF = {"cat2k": "icl-bc-passk-2k.json", "cat100": "icl-bc-passk-100.json"}.get(MODE, "icl-bc-passk.json")
MLPG = {"cat2k": "icl-tf-dbg-mlp2k-s*", "cat100": "icl-tf-dbg-mlp100-s*"}.get(MODE, "icl-norm-pose_sum*-s*")
NDEMO = {"cat2k": 2000, "cat100": 100}.get(MODE, 500)
NS = [100, 1000, 10000]; KS = np.arange(1, 9)
DIRK = lambda k: (DIR or "icl-tf-{k}").format(k=k)
R = {k: [json.load(open(p)) for p in sorted(glob.glob(f"{LOG_ROOT}/runs/{DIRK(k)}-s*/results.json"))] for k in ("fcfc", "fcfc_hist", "hist_c", "hist_c_noisy")}
RB = [json.load(open(p)) for p in sorted(glob.glob(f"{LOG_ROOT}/runs/" + ("icl-tf-dbg-c100-cbias" if MODE == "cat100" else "icl-tf-dbg-cbias2k") + "-s*/results.json"))]
BCK = json.load(open(f"{LOG_ROOT}/runs/{BCF}"))
pk = lambda k, n: np.array([r["runs"][f"{k}-N{n}"]["pass_k"] for r in R[k]])
def style(a, xl, yl, t):
    a.set_xlabel(xl); a.set_ylabel(yl); a.set_title(t, fontsize=11); a.grid(alpha=.25)
    for s in ("top", "right"): a.spines[s].set_visible(False)

# 1: T1 (3 blues by N) vs transformer BC
BLUES = {100: "#9cc3f0", 1000: "#4a90e2", 10000: "#12468f"}
fig, a = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
for n in NS:
    v = pk("fcfc", n); a.plot(KS, v.mean(0), "-o", color=BLUES[n], lw=2, ms=4, label=f"T1 [f,c,f+c], N={n} (hole hidden)")
    a.fill_between(KS, v.min(0), v.max(0), color=BLUES[n], alpha=.15)
for key, col, ls, lab in (("tfbc_visible", "#e8742a", "-", f"Transformer BC, hole visible ({NDEMO} demos)"), ("tfbc_hidden", "#8a4b12", "--", "Transformer BC, hole hidden")):
    v = np.array([x["pass_k"] for k, x in BCK.items() if k.endswith(key)])
    a.plot(KS, v.mean(0), ls, marker="s", color=col, lw=2, ms=4, label=lab)
    a.fill_between(KS, v.min(0), v.max(0), color=col, alpha=.15)
a.set_ylim(0, 1); a.legend(fontsize=8, loc="center right"); style(a, "k (attempts)", "pass@k", "T1 vs transformer BC (mean of 3 seeds, band = seed range)" + TTL)
fig.savefig(f"{OUT}/plot1_t1_vs_bc{SUF}.png", dpi=140)

# 2: pass@2 vs N, T1 vs MLP
mlp = {n: [] for n in NS}
for p in sorted(glob.glob(f"{LOG_ROOT}/runs/{MLPG}/results.json")):
    for n in NS:
        v = json.load(open(p))["runs"].get(f"N{n}")
        if v: mlp[n].append(v["pass2"])
fig, a = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
for lab, col, d in (("Transformer T1", "#2a78d6", {n: pk("fcfc", n)[:, 1] for n in NS}), ("MLP [f,c,f+c] (input-concat by construction)", "#52514e", mlp)):
    xs = [n for n in NS if len(d[n])]
    a.plot(xs, [np.mean(d[n]) for n in xs], "-o", color=col, lw=2, label=lab)
    for n in xs: a.scatter([n] * len(d[n]), d[n], color=col, s=14, alpha=.45)
a.set_xscale("log"); a.set_ylim(0, 1); a.set_xticks(NS); a.set_xticklabels(NS); a.legend(fontsize=9)
style(a, "N suboptimal BC rollouts", "pass@2", "pass@2 vs N (dots = seeds)" + TTL)
fig.savefig(f"{OUT}/plot2_pass2_vs_N{SUF}.png", dpi=140)

# 3: pass@k for T1, A, B, C per N
KIND = [("fcfc", "T1 [f,c,f+c]", "#2a78d6"), ("fcfc_hist", "A  +history", "#eb6834"), ("hist_c", "B  history + c", "#1baf7a"), ("hist_c_noisy", "C  s~U(.9,1.1)", "#e87ba4")]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True, constrained_layout=True)
for a, n in zip(ax, NS):
    for k, lab, col in KIND:
        v = pk(k, n); a.plot(KS, v.mean(0), "-o", color=col, lw=2, ms=4, label=lab); a.fill_between(KS, v.min(0), v.max(0), color=col, alpha=.15)
    if BIAS:
        v = np.array([r["runs"][f"hist_c_noisy-N{n}"]["pass_k"] for r in RB]); a.plot(KS, v.mean(0), "--o", color="#8a2b5c", lw=2, ms=4, label="C-biased  s~U(1.1,1.3)"); a.fill_between(KS, v.min(0), v.max(0), color="#8a2b5c", alpha=.15)
    a.set_ylim(0, 1); style(a, "k (attempts)", "pass@k", f"N={n}")
ax[1].legend(fontsize=8, loc="center right")
fig.suptitle("pass@k by variant (mean of 3 seeds, band = seed range)" + TTL)
fig.savefig(f"{OUT}/plot3_variants{SUF}.png", dpi=140)
