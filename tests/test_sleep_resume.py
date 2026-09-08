"""Resume from a checkpoint restores the wake/sleep state, not just the model.

Sep 6 2026: a full PVC crashed every writer and each pod restarted from step
0 because train.py could not resume (HF restores model/optimizer/RNG/step;
the SleepSampler and the trainer's phase bookkeeping were never saved).
These tests pin the round trip: a sampler rebuilt from state_dict() yields
the same indices as the original, the trainer-level helpers restore
phase_steps and the per-phase sleep length, and the Job-side helper picks
only complete checkpoints.
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_curriculum.sleep_sampler import SleepSampler  # noqa: E402
from src.utils.sleep_state import (  # noqa: E402
    COMPLETE_MARKER,
    SLEEP_STATE_FILE,
    apply_trainer_sleep_state,
    latest_resumable_checkpoint,
    load_sleep_state,
    save_sleep_state,
    training_already_complete,
    trainer_sleep_state,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "k8s"))
from resume_args import resume_overrides  # noqa: E402


class _Dataset:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"input_ids": [0, 10 + (i % 50), 2]}


def _sampler(strategy="strict", contextualize=False, n=100, seed=0, **kw):
    np.random.seed(seed)
    return SleepSampler(_Dataset(n), batch_size=10, replay_ratio=0.2, n_phases=5,
                        n_augmentations=3, decay_rate=0.05, min_decay_factor=0.2,
                        contextualize_sleep=contextualize, replay_strategy=strategy, **kw)


def _take(it, k):
    return [next(it) for _ in range(k)]


def _advance_two_cycles(s):
    it = iter(s)
    fold0 = _take(it, 20)
    s.add_to_candidates(fold0, [float(i) for i in fold0])
    s.switch_phase("SLEEP")
    _take(it, 7)
    s.switch_phase("WAKE")
    fold1 = _take(it, 13)
    s.add_to_candidates(fold1, [1.0] * 13)
    return it


def _roundtrip(s):
    state = s.state_dict()
    fresh = _sampler(strategy=s.replay_strategy, contextualize=s.contextualize_sleep, seed=123)
    assert fresh.folds != s.folds  # different shuffle before loading
    fresh.load_state_dict(state)
    return fresh


def test_wake_resume_continues_the_same_fold_at_the_same_position():
    s = _sampler()
    it = _advance_two_cycles(s)
    fresh = _roundtrip(s)
    assert fresh.folds == s.folds and fresh.curr_fold == s.curr_fold == 1
    assert fresh.wake_pos == s.wake_pos == 13
    assert fresh.wake_candidates == s.wake_candidates
    assert _take(iter(fresh), 20) == _take(it, 20)


def test_sleep_resume_without_contextualization_replays_identically():
    s = _sampler()
    it = _advance_two_cycles(s)
    s.switch_phase("SLEEP")
    _take(it, 5)
    fresh = _roundtrip(s)
    assert fresh.phase == "SLEEP" and fresh.replay_buffer == s.replay_buffer
    assert fresh.sleep_pos == s.sleep_pos == 5
    assert _take(iter(fresh), 30) == _take(it, 30)


def test_sleep_resume_with_contextualization_replays_the_same_buffer_in_a_new_order():
    s = _sampler(contextualize=True)
    it = _advance_two_cycles(s)
    s.switch_phase("SLEEP")
    _take(it, 4)
    fresh = _roundtrip(s)
    assert fresh.replay_buffer == s.replay_buffer
    assert len(fresh.contextualized_chunks) == len(s.contextualized_chunks) == 3 * len(s.replay_buffer)
    assert sorted(fresh.contextualized_chunks) == sorted(s.contextualized_chunks)
    # the next wake phase is unaffected by the re-shuffle
    fresh.switch_phase("WAKE")
    s.switch_phase("WAKE")
    assert _take(iter(fresh), 15) == _take(it, 15)


def test_random_strategy_buffer_survives_as_plain_ints(tmp_path):
    s = _sampler(strategy="random")
    _advance_two_cycles(s)
    s.switch_phase("SLEEP")
    state = s.state_dict()
    path = tmp_path / SLEEP_STATE_FILE
    save_sleep_state(str(path), {"sampler": state})
    loaded = load_sleep_state(str(path))["sampler"]
    fresh = _sampler(strategy="random", seed=9)
    fresh.load_state_dict(loaded)
    assert fresh.replay_buffer == [int(i) for i in s.replay_buffer]
    assert all(type(i) is int for i in fresh.replay_buffer)


def test_load_rejects_a_checkpoint_from_a_different_regime():
    s = _sampler()
    state = s.state_dict()
    other = SleepSampler(_Dataset(100), batch_size=10, replay_ratio=0.2, n_phases=4, n_augmentations=1)
    with pytest.raises(ValueError):
        other.load_state_dict(state)
    smaller = SleepSampler(_Dataset(90), batch_size=10, replay_ratio=0.2, n_phases=5, n_augmentations=1)
    with pytest.raises(ValueError):
        smaller.load_state_dict(state)


def test_trainer_state_round_trip_restores_phase_steps_and_sleep_length(tmp_path):
    trainer = SimpleNamespace(global_step=86_000, phase_steps=413,
                              max_steps_per_phase={"SLEEP": 1_927, "WAKE": 688},
                              _sh_time_sec={"per_sample_grad": 0.0, "score": 12.5, "shrink": 3.0})
    d = trainer_sleep_state(trainer, {"phase": "SLEEP"})
    path = tmp_path / SLEEP_STATE_FILE
    save_sleep_state(str(path), d)
    fresh = SimpleNamespace(global_step=0, phase_steps=0,
                            max_steps_per_phase={"SLEEP": 241, "WAKE": 688},
                            _sh_time_sec={"per_sample_grad": 0.0, "score": 0.0, "shrink": 0.0})
    loaded = load_sleep_state(str(path))
    apply_trainer_sleep_state(fresh, loaded)
    assert fresh.phase_steps == 413
    assert fresh.max_steps_per_phase == {"SLEEP": 1_927, "WAKE": 688}
    assert fresh._sh_time_sec["score"] == 12.5
    assert loaded["sampler"] == {"phase": "SLEEP"}


def test_latest_resumable_checkpoint_skips_incomplete_dirs(tmp_path):
    run = tmp_path / "run"
    for step, complete, with_state in ((100, True, True), (200, True, True), (300, False, True), (400, True, False)):
        d = run / f"checkpoint-{step}"
        d.mkdir(parents=True)
        if complete:
            (d / COMPLETE_MARKER).write_text("x")
        if with_state:
            (d / SLEEP_STATE_FILE).write_text("x")
    assert latest_resumable_checkpoint(str(run)) == str(run / "checkpoint-200")
    assert latest_resumable_checkpoint(str(tmp_path / "missing")) is None


def test_resume_overrides_need_run_id_and_a_complete_checkpoint(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoint-50").mkdir(parents=True)
    assert resume_overrides(str(run)) == ""
    (run / "wandb_run_id.txt").write_text("abc123\n")
    assert resume_overrides(str(run)) == ""  # checkpoint incomplete
    (run / "checkpoint-50" / COMPLETE_MARKER).write_text("x")
    (run / "checkpoint-50" / SLEEP_STATE_FILE).write_text("x")
    expected = f"experiment.resume_checkpoint_path={run / 'checkpoint-50'} experiment.resume_run_id=abc123"
    assert resume_overrides(str(run)) == expected
    assert resume_overrides(str(run), auto_resume="0") == ""


def test_a_checkpoint_at_max_steps_means_training_is_complete(tmp_path):
    """Sep 7 2026: a container restarted after training finished resumed from
    the final checkpoint and trainer.train() raised on its first step, so the
    Job crash-looped. train.py must skip training in that case."""
    ck = tmp_path / "checkpoint-750"
    ck.mkdir()
    assert training_already_complete(str(ck), 750) is False  # no trainer_state.json yet
    (ck / "trainer_state.json").write_text(json.dumps({"global_step": 750}))
    assert training_already_complete(str(ck), 750) is True
    assert training_already_complete(str(ck), 751) is False
    (ck / "trainer_state.json").write_text(json.dumps({"global_step": 375}))
    assert training_already_complete(str(ck), 750) is False
