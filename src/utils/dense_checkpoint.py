"""Extra checkpoints at an explicit list of steps (trainer.dense_save_steps).

HF's save_steps is an interval and, with load_best_model_at_end, is tied to
eval_steps, so it cannot be lowered without dragging evaluation along. A
callback only ORs into control.should_save: the interval checkpoints are
untouched and an empty list changes nothing. Used by the age-of-acquisition
runs, which need log-spaced checkpoints before the first interval save.
"""
from typing import Iterable

from transformers import TrainerCallback


class DenseCheckpointCallback(TrainerCallback):
    def __init__(self, steps: Iterable[int]):
        self.steps = frozenset(int(s) for s in steps)
        if any(s <= 0 for s in self.steps):
            raise ValueError(f"dense_save_steps must be positive, got {sorted(self.steps)}")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.steps:
            control.should_save = True
        return control
