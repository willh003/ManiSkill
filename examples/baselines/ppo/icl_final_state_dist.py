"""Plot the ICL training-context distribution (BC final peg pose, correction to a random expert final peg) for several N.
Usage: python icl_final_state_dist.py rollouts.pt out.png N1,N2"""
import sys, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = torch.load(sys.argv[1]); ns = [int(x) for x in sys.argv[3].split(",")]
fp, ep, ok = d["F_peg"].numpy(), d["e_peg"].numpy(), d["F_succ"].numpy()
yaw = lambda q: 2 * np.arctan2(q[:, 3], q[:, 0])
rng = np.random.default_rng(0)
fig, ax = plt.subplots(len(ns), 4, figsize=(17, 4.2 * len(ns)))
for r, n in enumerate(ns):
    f, s = fp[:n], ok[:n]
    e = ep[rng.integers(0, len(ep), n)]  # random expert pairing
    c = e[:, :3] - f[:, :3]
    nrm = np.linalg.norm(c, axis=1)
    far = (np.abs(f[:, :2]) > 1).any(1)
    print(f"N={n}: BC success {s.mean():.3f}, final peg >1m from origin {far.mean():.3f}, |corr| median {np.median(nrm):.3f} p90 {np.quantile(nrm,.9):.3f} p99 {np.quantile(nrm,.99):.3f} max {nrm.max():.2f}; corr<5cm {np.mean(nrm<.05):.3f}")
    a = ax[r, 0]
    a.scatter(f[~s, 0], f[~s, 1], s=4 if n > 1000 else 14, c="C3", alpha=.35, label="BC fail")
    a.scatter(f[s, 0], f[s, 1], s=4 if n > 1000 else 14, c="C2", alpha=.35, label="BC success")
    a.scatter(ep[:, 0], ep[:, 1], s=6, c="k", marker="x", alpha=.5, label="expert final (500)")
    a.set_xlim(-.45, .45); a.set_ylim(-.3, .6); a.set_title(f"N={n}: final peg xy (zoomed; {far.mean()*100:.1f}% beyond 1m)"); a.legend(fontsize=7); a.set_aspect("equal")
    a = ax[r, 1]
    a.scatter(c[:, 0], c[:, 1], s=4 if n > 1000 else 14, c="C0", alpha=.35)
    a.set_xlim(-.6, .6); a.set_ylim(-.6, .6); a.set_title("correction xy (expert final - BC final)"); a.set_aspect("equal"); a.axhline(0, c="k", lw=.3); a.axvline(0, c="k", lw=.3)
    a = ax[r, 2]
    a.hist(np.log10(nrm + 1e-4), bins=60, color="C0"); a.set_title("log10 |correction| (m)"); a.set_xlabel("log10 m")
    a = ax[r, 3]
    a.hist(np.clip(yaw(e) - yaw(f), -3.2, 3.2), bins=60, color="C1"); a.set_title("yaw correction (rad, clipped)")
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=110)
