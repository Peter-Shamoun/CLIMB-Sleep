"""Sleep-mechanism state that HF checkpoints do not carry.

HF Trainer checkpoints restore model, optimizer, scheduler, RNG and
global_step. The wake/sleep bookkeeping (which phase, which fold, the
candidate pool and the replay buffer, steps into the current phase, the
per-phase sleep length) lives in CustomTrainer and SleepSampler, so without
this file a restarted pod trains from step 0 again (Sep 6 2026: ~120 GPU-hours
lost to one full PVC). CustomTrainer._save writes SLEEP_STATE_FILE next to
every checkpoint; _load_from_checkpoint reads it back.

The pure functions here take any object with the trainer attributes they
touch so they can be tested with a SimpleNamespace. Serialization is
torch.save of a plain dict of scalars and numpy arrays.
"""
import glob
import json
import os
import re
from typing import Any, Dict, Optional

import torch

SLEEP_STATE_FILE = "sleep_state.pt"
# HF writes rng_state.pth after the optimizer and scheduler in
# _save_checkpoint, so its presence means the checkpoint is complete.
COMPLETE_MARKER = "rng_state.pth"


def checkpoint_global_step(ckpt_dir: str) -> Optional[int]:
    """global_step recorded in HF's trainer_state.json inside a checkpoint."""
    p = os.path.join(ckpt_dir, "trainer_state.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        try:
            return int(json.load(f).get("global_step"))
        except (TypeError, ValueError):
            return None


def training_already_complete(ckpt_dir: str, max_steps: int) -> bool:
    """True when the checkpoint we would resume from already sits at
    max_steps. A container restarted after training finished (Sep 7 2026: the
    resume smoke crash-looped because the final checkpoint was picked up and
    trainer.train() raises on its first step) must skip training and go
    straight to the final evaluation."""
    step = checkpoint_global_step(ckpt_dir)
    return step is not None and step >= max_steps


def trainer_sleep_state(trainer: Any, sampler_state: Optional[dict]) -> Dict[str, Any]:
    return {
        "global_step": int(trainer.global_step),
        "phase_steps": int(trainer.phase_steps),
        "max_steps_per_phase": {k: int(v) for k, v in trainer.max_steps_per_phase.items()},
        "sh_time_sec": {k: float(v) for k, v in trainer._sh_time_sec.items()},
        "sampler": sampler_state,
    }


def apply_trainer_sleep_state(trainer: Any, state: Dict[str, Any]) -> None:
    """Restore the trainer-level fields. The sampler part is applied separately
    (the sampler may not exist yet when the checkpoint is loaded)."""
    trainer.phase_steps = int(state["phase_steps"])
    for k, v in state["max_steps_per_phase"].items():
        trainer.max_steps_per_phase[k] = int(v)
    for k, v in state.get("sh_time_sec", {}).items():
        trainer._sh_time_sec[k] = float(v)


def save_sleep_state(path: str, state: Dict[str, Any]) -> None:
    torch.save(state, path)


def load_sleep_state(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def latest_resumable_checkpoint(run_dir: str) -> Optional[str]:
    """Highest-numbered checkpoint-<step>/ that holds both the sleep state and
    HF's completion marker; None when there is nothing to resume from."""
    best_step, best_dir = -1, None
    for d in glob.glob(os.path.join(run_dir, "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", d)
        if not m:
            continue
        if not (
            os.path.exists(os.path.join(d, SLEEP_STATE_FILE))
            and os.path.exists(os.path.join(d, COMPLETE_MARKER))
        ):
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step, best_dir = step, d
    return best_dir
