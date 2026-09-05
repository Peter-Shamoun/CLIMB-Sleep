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
