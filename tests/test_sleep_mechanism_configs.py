"""Every conf/sleep_mechanism/*.yaml must use keys the sampler and trainer act on.

Guards against a silent failure mode: `replay_strategy: "utility"` matched no
branch in SleepSampler.update_replay_buffer, so the replay buffer stayed empty
and the sleep phase trained on nothing. The right split is
replay_criteria (loss | utility) x replay_strategy (strict | weighted | random).

Run: python -m pytest tests/ -v
"""

import glob
import os

import pytest
import yaml

CONF_DIR = os.path.join(os.path.dirname(__file__), "..", "conf", "sleep_mechanism")
CONFIGS = sorted(glob.glob(os.path.join(CONF_DIR, "*.yaml")))

# Mirror of the branches in SleepSampler.update_replay_buffer and the checks in
# CustomTrainer.compute_loss / __init__.
STRATEGIES = {"strict", "weighted", "random"}
CRITERIA = {"loss", "utility"}
SIGNALS = {"fisher", "taylor_signed", "taylor_abs"}
SCOPES = {"per_tensor", "global"}


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("path", CONFIGS, ids=os.path.basename)
def test_replay_strategy_is_one_the_sampler_handles(path):
    cfg = load(path)
    assert cfg.get("replay_strategy", "weighted") in STRATEGIES, (
        f"{os.path.basename(path)}: replay_strategy must be one of "
        f"{sorted(STRATEGIES)}; 'utility' belongs in replay_criteria"
    )


@pytest.mark.parametrize("path", CONFIGS, ids=os.path.basename)
def test_replay_criteria_is_valid(path):
    cfg = load(path)
    assert cfg.get("replay_criteria", "loss") in CRITERIA


@pytest.mark.parametrize("path", CONFIGS, ids=os.path.basename)
def test_plasticity_decay_block_is_well_formed(path):
    cfg = load(path)
    pd = cfg.get("plasticity_decay")
    if pd is None:
        return
    assert pd.get("decay_type", "fisher_protected_shrink") == "fisher_protected_shrink"
    assert pd.get("importance_signal", "fisher") in SIGNALS
    assert 0.0 < pd.get("shrink_factor", 0.95) <= 1.0
    assert 0.0 <= pd.get("protect_top_fraction", 0.2) <= 1.0
    assert pd.get("threshold_scope", "per_tensor") in SCOPES


def test_sh_arms_share_the_replay_regime():
    """sh_off / sh_taylor_signed / sh_taylor_abs / sh_fisher differ only in SH."""
    arms = {
        n: load(os.path.join(CONF_DIR, f"{n}.yaml"))
        for n in ("sh_off", "sh_taylor_signed", "sh_taylor_abs", "sh_fisher")
    }
    control = {k: v for k, v in arms["sh_off"].items() if k != "plasticity_decay"}
    for name, cfg in arms.items():
        replay = {k: v for k, v in cfg.items() if k != "plasticity_decay"}
        assert replay == control, f"{name} drifted from sh_off on {set(replay) ^ set(control) or 'values'}"
    assert "plasticity_decay" not in arms["sh_off"]
    for name in ("sh_taylor_signed", "sh_taylor_abs", "sh_fisher"):
        assert arms[name]["plasticity_decay"]["importance_signal"] == name[len("sh_"):]


def _train_py_step_budget(cfg, total_wake_steps):
    """Replicates the step-budget formula in train.py (sleep_wake_ratio > 0)."""
    import math

    n_phases = cfg["n_phases"]
    total_wake_steps = min(cfg["wake_block_steps"] * n_phases, total_wake_steps)
    wake_per_phase = math.ceil(total_wake_steps / n_phases)
    sleep_per_phase = int(wake_per_phase * cfg["sleep_wake_ratio"])
    return int(total_wake_steps + sleep_per_phase * n_phases)


def test_baseline_like_clm_matches_sleep_cell_budget():
    """The CLM epoch control trains for the same 10-epoch budget as the sh_* cells."""
    # strict 100M at batch 32: ceil(len(train) / 32) = 34,394 wake steps
    # (train/global_step 343,994 on the sh-arms-clm and rply_expmt_*_0.1 runs).
    total_wake = 34394
    base = load(os.path.join(CONF_DIR, "baseline_like_clm.yaml"))
    sleep = load(os.path.join(CONF_DIR, "sh_off.yaml"))
    assert _train_py_step_budget(sleep, total_wake) == 343994
    assert _train_py_step_budget(base, total_wake) == 343940
    # Only per-phase ceil rounding separates the two budgets (< 0.02%).
    assert abs(343994 - 343940) / 343994 < 2e-4
    # Epoch semantics: one phase, replay everything, no contextualization.
    assert base["n_phases"] == 1
    assert base["replay_ratio"] == 1.0
    assert base["contextualize_sleep"] is False
    assert base["sleep_wake_ratio"] == sleep["sleep_wake_ratio"] == 9.0
