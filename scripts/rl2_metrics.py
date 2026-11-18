"""rl2_peg run dirs -> tracker docs. Usage: python scripts/rl2_metrics.py OUT_DIR run_dir [...]
Writes OUT_DIR/<run-id>.metrics.json ({series}) and OUT_DIR/<run-id>.last.json ({lastMetrics, heartbeat, updated})."""
import datetime, json, os, sys

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
KEYS = ["eval_succ_ep1", "eval_succ_ep2", "eval_succ_ep3", "eval_pass@2", "eval_pass@3", "train_succ_ep1", "train_succ_ep2", "train_succ_ep3",
        "bc_loss", "label_mse_ep2", "vf", "kl", "logstd"]
for d in sys.argv[2:]:
    rid = os.path.basename(d.rstrip("/"))
    S, last = {}, {}
    p = f"{d}/metrics.jsonl"
    if not os.path.exists(p):
        continue
    for line in open(p):
        m = json.loads(line)
        for k in KEYS:
            if k in m:
                S.setdefault(k, []).append([m["step"], round(m[k], 5)])
                last[k] = round(m[k], 4)
        last["step"] = m["step"]
    json.dump(dict(series=S), open(f"{out}/{rid}.metrics.json", "w"))
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    json.dump(dict(lastMetrics=last, heartbeat=now, updated=now), open(f"{out}/{rid}.last.json", "w"))
    print(rid, {k: last.get(k) for k in ("step", "eval_succ_ep1", "eval_succ_ep2", "eval_succ_ep3")})
