"""perf@k (success in episode k): recurrent PPO no-noise vs hidden-f 4cm vs standard PPO baseline. Usage: python rl2_hf_compare_plot.py OUT.png"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
P = "logs/klone-pull/"
perf = lambda n, f="it170.json": json.load(open(f"{P}rl2-peg-e8-{n}/{f}"))["perf"]
fin = np.array(json.load(open(P + "rl2-peg-e8-n4-hf-s1/results.json"))["succ_matrix"]).mean(0)
ks = range(1, 9)
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.axhline(0.988, color="#8C8C8C", ls="--", lw=2, label="standard PPO baseline (full obs, 1 ep): 0.99")
ax.plot(ks, perf("n0-s1", "it180.json"), color="#124690", lw=2, marker="o", label="recurrent PPO, no noise (latest ckpt, it180)")
ax.plot(ks, fin, color="#B84A0A", lw=2, marker="o", label="recurrent PPO, hidden f, 4 cm (it300, final)")
ax.set_xlabel("k (episode)"); ax.set_ylabel("perf@k (success in episode k)"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(loc="lower right", fontsize=7.5)
ax.set_title("8-episode recurrent PPO (seed 1, 512 test tasks)", fontsize=10)
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=130)
