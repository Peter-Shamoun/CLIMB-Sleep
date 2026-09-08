"""Step budget for wake/sleep training, shared by train.py, the trainer and tests.

Two ways to size a sleep phase:

* fixed length (``replay_repeats <= 0``, the historical behaviour):
  ``sleep_steps_per_phase = wake_steps_per_phase * sleep_wake_ratio`` (or the
  remainder of ``max_training_steps`` split evenly when the ratio is <= 0).
  The replay buffer grows with the number of folds seen, so each buffer
  sample is replayed ``sleep_samples / buffer`` times: 90/k in cycle k for the
  ratio-0.1 cells (see paper_results.md, Experiment 5, "over-replay").

* bounded repeats (``replay_repeats > 0``): sleep phase k lasts
  ``ceil(replay_repeats * B_k / batch)`` steps with ``B_k`` the buffer size in
  cycle k, so every buffer sample is replayed ``replay_repeats`` times in every
  cycle. The trainer sets the per-phase length at each WAKE->SLEEP switch from
  the realized buffer; the total budget is computed up front from the same
  formula so ``max_steps`` matches (up to the trainer wrapping into fold 0 for
  the last few steps if the realized total falls short).
"""
import math
from typing import Dict


def wake_samples_per_phase(wake_steps_per_phase: int, batch_size: int, fold_size: int) -> int:
    """Distinct samples a wake phase draws from its fold (no wrap-around)."""
    return min(wake_steps_per_phase * batch_size, fold_size)


def buffer_size(k: int, wake_samples: int, replay_ratio: float) -> int:
    """Replay buffer after k wake phases: SleepSampler keeps every seen sample
    as a candidate and takes ``int(len(candidates) * replay_ratio)``."""
    return int(k * wake_samples * replay_ratio)


def sleep_steps_for_buffer(n_buffer: int, replay_repeats: float, batch_size: int) -> int:
    """Steps needed to replay each of ``n_buffer`` samples ``replay_repeats`` times."""
    if n_buffer <= 0:
        return 0
    return math.ceil(replay_repeats * n_buffer / batch_size)


def total_sleep_steps(
    n_phases: int, wake_samples: int, replay_ratio: float, replay_repeats: float, batch_size: int
) -> int:
    return sum(
        sleep_steps_for_buffer(buffer_size(k, wake_samples, replay_ratio), replay_repeats, batch_size)
        for k in range(1, n_phases + 1)
    )


def step_budget(
    n_train: int,
    batch_size: int,
    n_phases: int,
    wake_block_steps: int,
    sleep_wake_ratio: float,
    max_training_steps: int,
    replay_ratio: float = 0.1,
    replay_repeats: float = -1.0,
) -> Dict[str, int]:
    """Mirror of the budget logic in train.py. Returns integers:
    total_wake_steps, wake_steps_per_phase, sleep_steps_per_phase (first sleep
    phase under bounded repeats), total_sleep_steps, max_training_steps.
    """
    total_wake_steps = min(wake_block_steps * n_phases, math.ceil(n_train / batch_size))
    wake_steps_per_phase = math.ceil(total_wake_steps / n_phases)

    if replay_repeats > 0:
        fold_size = n_train // n_phases
        wake_samples = wake_samples_per_phase(wake_steps_per_phase, batch_size, fold_size)
        sleep_total = total_sleep_steps(n_phases, wake_samples, replay_ratio, replay_repeats, batch_size)
        sleep_steps_per_phase = sleep_steps_for_buffer(
            buffer_size(1, wake_samples, replay_ratio), replay_repeats, batch_size
        )
        max_steps = total_wake_steps + sleep_total
    elif sleep_wake_ratio > 0:
        sleep_steps_per_phase = wake_steps_per_phase * sleep_wake_ratio
        sleep_total = sleep_steps_per_phase * n_phases
        max_steps = total_wake_steps + sleep_total
    else:
        sleep_total = max_training_steps - total_wake_steps
        sleep_steps_per_phase = math.ceil(sleep_total / n_phases)
        max_steps = max_training_steps

    return {
        "total_wake_steps": int(total_wake_steps),
        "wake_steps_per_phase": int(wake_steps_per_phase),
        "sleep_steps_per_phase": int(sleep_steps_per_phase),
        "total_sleep_steps": int(sleep_total),
        "max_training_steps": int(max_steps),
    }
