"""Uniform run directory for experiment scripts.

    <out_dir>/
        config.json          args, git commit + dirty flag, command line, Slurm job id/name and its log file
        train.log            copy of stdout/stderr (the Slurm log in logs/ has the same text plus scheduler noise)
        metrics.jsonl        one JSON object per logged step
        tb/                  TensorBoard events with the same scalars
        checkpoints/         model weights
        results.json         final numbers (references, controls, ...)
        plots/               figures
"""
import dataclasses
import json
import os
import subprocess
import sys
import time


class _Tee:
    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, "a", buffering=1)

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _git(*cmd):
    try:
        return subprocess.run(["git", *cmd], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


class RunDir:
    def __init__(self, path, args):
        self.path = path
        for sub in ("checkpoints", "plots", "tb"):
            os.makedirs(f"{path}/{sub}", exist_ok=True)
        sys.stdout, sys.stderr = _Tee(sys.stdout, f"{path}/train.log"), _Tee(sys.stderr, f"{path}/train.log")
        job, name = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_JOB_NAME")
        self.save_json("config.json", dict(
            args=dataclasses.asdict(args) if dataclasses.is_dataclass(args) else vars(args),
            command=" ".join(sys.argv), started=time.strftime("%Y-%m-%d %H:%M:%S"),
            git_commit=_git("rev-parse", "HEAD"), git_dirty=bool(_git("status", "--porcelain", "--untracked-files=no")),
            slurm=dict(job_id=job, job_name=name, log_file=f"logs/{name}-{job}.out" if job else None),
        ))
        from torch.utils.tensorboard import SummaryWriter

        self.tb = SummaryWriter(f"{path}/tb")

    def log(self, step, metrics):
        """metrics: flat dict of floats."""
        with open(f"{self.path}/metrics.jsonl", "a") as f:
            f.write(json.dumps(dict(step=step, **metrics)) + "\n")
        for k, v in metrics.items():
            self.tb.add_scalar(k, v, step)
        self.tb.flush()

    def save_json(self, name, obj):
        with open(f"{self.path}/{name}", "w") as f:
            json.dump(obj, f, indent=1)

    def ckpt(self, name):
        return f"{self.path}/checkpoints/{name}"

    def plot(self, name):
        return f"{self.path}/plots/{name}"
