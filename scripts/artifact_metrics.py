"""Convert icl_final_state run dirs into ArtifactData docs: metrics/<run-id> {series:{...}} and lastMetrics.
Usage: python scripts/artifact_metrics.py OUT_DIR run_dir [run_dir ...]   (writes OUT_DIR/<run-id>.metrics.json and OUT_DIR/<run-id>.last.json)"""
import datetime, json, os, sys
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for d in sys.argv[2:]:
    rid = os.path.basename(d.rstrip("/"))
    S, last = {}, {}
    tp = f"{d}/trace.json"
    if os.path.exists(tp):
        t = json.load(open(tp))
        for k, v in t.items():
            if k == "bc":
                S["bc_success"] = [[x[0], x[1]] for x in v]
                if v: last["bc_success"] = v[-1][1]
            else:
                S[f"pass2_{k}"] = [[x[0], x[2]] for x in v]
                S[f"val_mse_{k}"] = [[x[0], x[4]] for x in v]
                if v: last[f"pass2_{k}"] = v[-1][2]; last[f"step_{k}"] = v[-1][0]
    cp = f"{d}/curves.json"
    if os.path.exists(cp):
        c = json.load(open(cp))
        for k, v in c.items():
            S[f"train_loss_{k}"] = [[x[0], x[1]] for x in v[::5]]
            S[f"val_loss_{k}"] = [[x[0], x[3]] for x in v[::5]]
    json.dump(dict(series=S), open(f"{out}/{rid}.metrics.json", "w"))
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    json.dump(dict(lastMetrics=last, heartbeat=now, updated=now), open(f"{out}/{rid}.last.json", "w"))
    print(rid, last)
