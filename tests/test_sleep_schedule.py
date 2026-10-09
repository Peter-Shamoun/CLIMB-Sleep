"""Bounded-repeat sleep schedule (replay_repeats > 0).

Experiment 5 (paper_results.md) attributed random > loss under CLM to
over-replay: the sleep length is fixed while the buffer grows with the folds
seen, so each buffer sample is replayed 90/k times in cycle k. These tests pin
the alternative schedule down: sleep phase k lasts ceil(repeats * B_k / batch)
steps, the total budget is computed from the same formula so max_steps matches,
and replay_repeats <= 0 reproduces the historical 343,994-step budget exactly.
"""
import math
import os
import sys
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.sleep_schedule import (  # noqa: E402
    buffer_size,
    sleep_steps_for_buffer,
    step_budget,
    total_sleep_steps,
    wake_samples_per_phase,
)

# strict 100M at batch 32 on main_refactor (Sep 2026): 1,100,608 packed
# sequences, 50 folds of 22,012, 688 wake steps per phase. 688 x 32 = 22,016
# draws per wake phase, so the sampler wraps 4 indices and every fold sample
# enters the candidate pool: the buffer in cycle k is int(k * 22,012 * 0.1).
N_TRAIN, BATCH, N_PHASES, RATIO, REPEATS = 1_100_608, 32, 50, 0.1, 3.5
WAKE_SAMPLES = 22_012
CONF_DIR = os.path.join(os.path.dirname(__file__), "..", "conf", "sleep_mechanism")


def _load(name):
    with open(os.path.join(CONF_DIR, f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def test_wake_samples_and_first_buffer_match_the_production_cell():
    fold = N_TRAIN // N_PHASES
    assert fold == 22_012
    w = wake_samples_per_phase(688, BATCH, fold)
    assert 688 * BATCH == 22_016 > fold  # wraps: whole fold seen, 4 twice
    assert w == WAKE_SAMPLES
    assert buffer_size(1, w, RATIO) == 2_201
    assert buffer_size(50, w, RATIO) == 110_060


def test_sleep_steps_per_phase_bound_the_repeats():
    assert sleep_steps_for_buffer(2_201, REPEATS, BATCH) == 241  # ceil(240.7)
    assert sleep_steps_for_buffer(110_060, REPEATS, BATCH) == 12_038
    assert sleep_steps_for_buffer(0, REPEATS, BATCH) == 0
    # realized repeats per sample never fall below the target
    for n in (1, 31, 32, 33, 2_201, 110_060):
        steps = sleep_steps_for_buffer(n, REPEATS, BATCH)
        assert steps * BATCH / n >= REPEATS
        assert steps * BATCH / n < REPEATS + BATCH / n


def test_total_sleep_steps_is_the_per_phase_sum():
    w = WAKE_SAMPLES
    explicit = sum(math.ceil(REPEATS * int(k * w * RATIO) / BATCH) for k in range(1, N_PHASES + 1))
    assert total_sleep_steps(N_PHASES, w, RATIO, REPEATS, BATCH) == explicit


def test_fixed_length_budget_reproduces_the_historical_step_count():
    cfg = _load("sh_off")
    b = step_budget(N_TRAIN, BATCH, cfg["n_phases"], cfg["wake_block_steps"],
                    cfg["sleep_wake_ratio"], 400_000, cfg["replay_ratio"], -1.0)
    assert b["total_wake_steps"] == 34_394
    assert b["wake_steps_per_phase"] == 688
    assert b["sleep_steps_per_phase"] == 6_192
    assert b["max_training_steps"] == 343_994
    base = _load("baseline_like_clm")
    assert step_budget(N_TRAIN, BATCH, base["n_phases"], base["wake_block_steps"],
                       base["sleep_wake_ratio"], 400_000, base["replay_ratio"])["max_training_steps"] == 343_940


def test_max_steps_fallback_when_no_ratio_is_given():
    b = step_budget(N_TRAIN, BATCH, N_PHASES, 9_999_999, -1.0, 100_000, RATIO)
    assert b["max_training_steps"] == 100_000
    assert b["sleep_steps_per_phase"] == math.ceil((100_000 - 34_394) / N_PHASES)


def test_bounded_repeat_budget_matches_the_fixed_cells_within_one_percent():
    for name in ("sh_rr_strict", "sh_rr_random"):
        cfg = _load(name)
        assert cfg["replay_repeats"] == REPEATS
        b = step_budget(N_TRAIN, BATCH, cfg["n_phases"], cfg["wake_block_steps"],
                        cfg["sleep_wake_ratio"], 400_000, cfg["replay_ratio"], cfg["replay_repeats"])
        assert b["total_wake_steps"] == 34_394
        assert b["sleep_steps_per_phase"] == 241  # first sleep phase
        assert abs(b["max_training_steps"] - 343_994) / 343_994 < 0.01
        assert b["total_sleep_steps"] == total_sleep_steps(N_PHASES, WAKE_SAMPLES, RATIO, REPEATS, BATCH)


def test_rr_configs_equal_their_fixed_length_bases_except_replay_repeats():
    for rr, base in (("sh_rr_strict", "sh_off"), ("sh_rr_random", "sh_rand_off")):
        rr_cfg = _load(rr)
        assert {k: v for k, v in rr_cfg.items() if k != "replay_repeats"} == _load(base), rr


def test_trainer_resizes_the_sleep_phase_from_the_realized_buffer():
    pytest.importorskip("datasets")  # src.trainer pulls in the HF stack
    from src.trainer import CustomTrainer

    t = CustomTrainer.__new__(CustomTrainer)  # no model / args needed for this branch
    t.max_steps_per_phase = {"SLEEP": 6_192, "WAKE": 688}
    t._sleep_batch_size = BATCH
    t.replay_repeats = REPEATS
    assert t._sleep_steps_for_next_phase(SimpleNamespace(replay_buffer=list(range(2_201)))) == 241
    assert t._sleep_steps_for_next_phase(SimpleNamespace(replay_buffer=list(range(110_060)))) == 12_038
    t.replay_repeats = -1.0
    assert t._sleep_steps_for_next_phase(SimpleNamespace(replay_buffer=list(range(2_201)))) == 6_192
