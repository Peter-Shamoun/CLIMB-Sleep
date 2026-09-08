#!/usr/bin/env python3
"""Print the Hydra overrides that resume a run from its last complete checkpoint.

Usage (from the CLIMB-Sleep root, inside a Job container):
    RESUME_ARGS=$(python scripts/k8s/resume_args.py <run_dir>)
    python train.py ... $RESUME_ARGS

Prints nothing (fresh start) when AUTO_RESUME is "0", the run dir has no
wandb_run_id.txt, or no checkpoint-<step>/ holds both sleep_state.pt and
rng_state.pth. Otherwise prints
    experiment.resume_checkpoint_path=<dir> experiment.resume_run_id=<id>
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.utils.sleep_state import latest_resumable_checkpoint  # noqa: E402


def resume_overrides(run_dir: str, auto_resume: str = "1") -> str:
    if auto_resume != "1":
        return ""
    id_file = os.path.join(run_dir, "wandb_run_id.txt")
    if not os.path.isfile(id_file):
        return ""
    ckpt = latest_resumable_checkpoint(run_dir)
    if ckpt is None:
        return ""
    with open(id_file) as f:
        run_id = f.read().strip()
    if not run_id:
        return ""
    return f"experiment.resume_checkpoint_path={ckpt} experiment.resume_run_id={run_id}"


if __name__ == "__main__":
    print(resume_overrides(sys.argv[1], os.environ.get("AUTO_RESUME", "1")))
