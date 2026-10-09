"""Fixed-capacity memory stream (SleepSampler.memory_stream_capacity).

At every WAKE->SLEEP switch the stream keeps the top-capacity candidates by
score x decay and deletes the rest permanently; -1 leaves the historical
unbounded stream untouched.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_curriculum.sleep_sampler import SleepSampler  # noqa: E402
from src.utils.sleep_schedule import memory_stream_size  # noqa: E402


class _Dataset:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"input_ids": [0, 10 + (i % 50), 2]}


def _sampler(capacity=-1, n=100, strategy="strict"):
    np.random.seed(0)
    return SleepSampler(_Dataset(n), batch_size=10, replay_ratio=0.2, n_phases=5,
                        n_augmentations=3, decay_rate=0.5, min_decay_factor=0.2,
                        contextualize_sleep=False, replay_strategy=strategy,
                        memory_stream_capacity=capacity)


def _run_wake(s, scores):
    """Add the whole current fold with the given per-position scores."""
    fold = s.folds[s.curr_fold]
    s.add_to_candidates(fold, scores[: len(fold)])
    return fold


def _cycle(s, scores):
    fold = _run_wake(s, scores)
    s.switch_phase("SLEEP")
    s.switch_phase("WAKE")
    return fold


def test_memory_stream_size():
    assert memory_stream_size(-1, 1000) == -1
    assert memory_stream_size(0, 1000) == -1
    assert memory_stream_size(0.25, 1000) == 250
    assert memory_stream_size(1e-9, 1000) == 1
    with pytest.raises(ValueError):
        memory_stream_size(1.5, 1000)


def test_unbounded_keeps_everything():
    s = _sampler(capacity=-1)
    for _ in range(3):
        _cycle(s, [1.0] * 20)
    assert len(s.wake_candidates) == 60
    assert s.total_evicted == 0


def test_stream_never_exceeds_capacity_and_evictions_are_counted():
    s = _sampler(capacity=30)
    for k in range(4):
        fold = _run_wake(s, list(np.linspace(1, 2, 20)))
        s.switch_phase("SLEEP")
        assert len(s.wake_candidates) == min(30, 20 * (k + 1))
        assert len(s.replay_buffer) == int(len(s.wake_candidates) * 0.2)
        assert set(s.replay_buffer) <= set(s.wake_candidates)
        s.switch_phase("WAKE")
    assert s.total_evicted == 80 - 30


def test_eviction_keeps_highest_score_times_decay():
    # Old raw 3.0 decays to 1.5: ranking by raw score would keep the old fold,
    # ranking by score x decay keeps the 10 new samples at 2.0.
    s = _sampler(capacity=10)
    old = _cycle(s, [3.0] * 20)
    new = _run_wake(s, [1.0] * 10 + [2.0] * 10)
    s.switch_phase("SLEEP")
    assert set(s.wake_candidates) == set(new[10:])
    assert not (set(s.wake_candidates) & set(old))


def test_evicted_samples_are_gone_for_good():
    s = _sampler(capacity=20)
    first = _cycle(s, [0.1] * 20)        # low scores: evicted at the next switch
    _cycle(s, [5.0] * 20)
    for _ in range(2):
        _cycle(s, [0.01] * 20)
    assert not (set(first) & set(s.wake_candidates))


def test_random_strategy_draws_only_from_the_capped_stream():
    s = _sampler(capacity=25, strategy="random")
    for _ in range(3):
        _run_wake(s, list(np.random.rand(20)))
        s.switch_phase("SLEEP")
        assert set(s.replay_buffer) <= set(s.wake_candidates)
        s.switch_phase("WAKE")


def test_state_dict_round_trip_keeps_eviction_count():
    s = _sampler(capacity=25)
    for _ in range(3):
        _cycle(s, list(np.random.rand(20)))
    t = _sampler(capacity=25)
    t.load_state_dict(s.state_dict())
    assert t.total_evicted == s.total_evicted == 35
    assert t.wake_candidates == s.wake_candidates
