"""trainer.dense_save_steps: extra checkpoints at an explicit list of steps.

The age-of-acquisition analysis needs log-spaced checkpoints before the first
interval save (max_steps // 8 = 42,999, already 125M words). These tests drive
HF's own DefaultFlowCallback next to DenseCheckpointCallback, the way
CallbackHandler does, and record the steps at which should_save is raised.
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import TrainerControl, TrainerState  # noqa: E402
from transformers.trainer_callback import DefaultFlowCallback  # noqa: E402

from src.utils.dense_checkpoint import DenseCheckpointCallback  # noqa: E402

AOA_STEPS = [100, 250, 450, 800, 1400, 2500, 4500, 8000, 14000, 25000]


def _saved_steps(dense, max_steps=4_000, save_steps=500):
    args = SimpleNamespace(logging_strategy="no", eval_strategy="steps", save_strategy="steps",
                           eval_delay=0, logging_first_step=False)
    state = TrainerState(max_steps=max_steps, save_steps=save_steps, eval_steps=save_steps, logging_steps=10**9)
    callbacks = [DefaultFlowCallback(), DenseCheckpointCallback(dense)]
    saved = []
    for step in range(1, max_steps + 1):
        state.global_step = step
        control = TrainerControl()  # HF resets should_save at every step begin
        for cb in callbacks:
            control = cb.on_step_end(args, state, control) or control
        if control.should_save:
            saved.append(step)
    return saved


def test_an_empty_list_leaves_the_interval_checkpoints_unchanged():
    assert _saved_steps([]) == list(range(500, 4_001, 500))


def test_dense_steps_are_saved_exactly_and_the_interval_cadence_still_fires():
    dense = [100, 250, 450, 800, 1400, 2500]
    assert _saved_steps(dense) == sorted(set(dense) | set(range(500, 4_001, 500)))


def test_a_dense_step_that_coincides_with_an_interval_save_is_saved_once():
    assert _saved_steps([500, 750]).count(500) == 1


def test_the_aoa_grid_survives_the_100_step_collapse_in_aoa_prepare():
    steps = sorted(DenseCheckpointCallback(AOA_STEPS).steps)
    assert steps == AOA_STEPS
    assert min(b - a for a, b in zip(steps, steps[1:])) >= 100


def test_hydra_list_values_and_bad_steps():
    from omegaconf import OmegaConf

    assert DenseCheckpointCallback(OmegaConf.create([100, 250])).steps == {100, 250}
    with pytest.raises(ValueError):
        DenseCheckpointCallback([0, 100])


def test_aoa_staging_takes_the_tokenizer_from_the_checkpoint_of_an_unfinished_run(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "eval"))
    import aoa_prepare

    run = tmp_path / "run"
    for step in (100, 250, 42_992, 43_000):
        ck = run / f"checkpoint-{step}"
        (ck / "lm_model").mkdir(parents=True)
        (ck / "model.safetensors").write_text("w")
        (ck / "config.json").write_text("{}")
        (ck / "lm_model" / "tokenizer.json").write_text(f"tok-{step}")
    entries = aoa_prepare.stage(str(run), str(tmp_path / "stage"), 1.409, 32, 128)
    assert [e["step"] for e in entries] == [100, 250, 43_000]  # <100-step neighbours collapse
    assert (tmp_path / "stage" / "100" / "tokenizer.json").read_text() == "tok-100"
    (run / "lm_model").mkdir()
    (run / "lm_model" / "tokenizer.json").write_text("tok-final")
    aoa_prepare.stage(str(run), str(tmp_path / "stage2"), 1.409, 32, 128)
    assert (tmp_path / "stage2" / "100" / "tokenizer.json").read_text() == "tok-final"
