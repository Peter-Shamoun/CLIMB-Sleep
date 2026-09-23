"""replay_rescore: re-measure every replay candidate's loss before selection.

The stale-score test for the random-over-strict gap under CLM (paper_results.md
Experiment 9). These tests pin down that the rescoring pass computes the same
per-sample score as the wake path, maps it back to the right index, leaves the
decay factors alone, and that strict selection then follows the fresh scores.
"""
import os
import sys

import pytest
import torch
import yaml
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_curriculum.sleep_sampler import SleepSampler  # noqa: E402
from src.utils.replay_score import per_sample_mean_token_loss, rescore_losses  # noqa: E402

PAD = 1
VOCAB = 32
SEQ = 8


class ToyCLM(nn.Module):
    """Returns (logits,) like the HF models; dropout makes eval mode observable."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(VOCAB, 16)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(16, VOCAB)

    def forward(self, input_ids, attention_mask=None):
        return (self.out(self.drop(self.emb(input_ids))),)


class ListDataset:
    """dataset[list_of_ints] -> {"input_ids": [...]}, as a HF Dataset does."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return {"input_ids": [self.rows[i] for i in idx]}
        return {"input_ids": self.rows[idx]}


def collate(examples):
    """Minimal stand-in for DataCollatorForLanguageModeling(mlm=False)."""
    ids = torch.tensor([e["input_ids"] for e in examples])
    labels = ids.clone()
    labels[ids == PAD] = -100
    return {"input_ids": ids, "attention_mask": (ids != PAD).long(), "labels": labels}


@pytest.fixture
def rows():
    g = torch.Generator().manual_seed(1)
    rows = torch.randint(2, VOCAB, (40, SEQ), generator=g).tolist()
    rows[3][5:] = [PAD] * (SEQ - 5)  # one padded row: pads must not count
    return rows


def _reference(model, rows, i):
    model.eval()
    with torch.no_grad():
        b = collate([{"input_ids": rows[i]}])
        logits = model(b["input_ids"])[0].transpose(-1, -2)
        return per_sample_mean_token_loss(logits[:, :, :-1], b["labels"][:, 1:]).item()


def test_rescore_matches_the_wake_score_and_maps_indices(rows):
    model = ToyCLM()
    idx = [31, 3, 0, 17, 9, 25]
    got = rescore_losses(model, ListDataset(rows), idx, collate, "clm", "cpu", batch_size=4)
    assert set(got) == set(idx)
    for i in idx:
        assert got[i] == pytest.approx(_reference(model, rows, i), rel=1e-5)


def test_rescore_runs_in_eval_mode_and_restores_training_mode(rows):
    model = ToyCLM()
    model.train()
    a = rescore_losses(model, ListDataset(rows), [5, 6], collate, "clm", "cpu")
    b = rescore_losses(model, ListDataset(rows), [5, 6], collate, "clm", "cpu")
    assert a == b  # dropout off, so deterministic
    assert model.training


def test_rescore_refuses_mlm(rows):
    with pytest.raises(ValueError):
        rescore_losses(ToyCLM(), ListDataset(rows), [0], collate, "mlm", "cpu")


def _sampler(n=100, ratio=0.1):
    s = SleepSampler(list(range(n)), batch_size=4, replay_ratio=ratio, n_phases=1,
                     replay_strategy="strict", contextualize_sleep=False)
    return s


def test_refresh_keeps_decay_factors_and_strict_follows_fresh_scores():
    s = _sampler()
    idx = list(range(100))
    s.add_to_candidates(idx, [float(i) for i in idx])  # stale: 90..99 on top
    s.wake_candidates = {i: (sc, 0.5 if i % 2 else 1.0) for i, (sc, _) in s.wake_candidates.items()}
    fresh = {i: float(100 - i) for i in idx}  # fresh: 0..9 on top
    diag = s.refresh_scores(fresh)
    assert all(s.wake_candidates[i] == (float(100 - i), 0.5 if i % 2 else 1.0) for i in idx)
    assert diag["rescore/n"] == 100
    assert diag["rescore/topk_overlap"] == 0.0
    assert diag["rescore/pearson"] == pytest.approx(-1.0)
    s.switch_phase("SLEEP")
    # strict ranks score x decay: the even indices among the lowest ids win
    expect = sorted(idx, key=lambda i: -(100 - i) * (0.5 if i % 2 else 1.0))[:10]
    assert sorted(s.replay_buffer) == sorted(expect)


def test_refresh_with_identical_scores_is_a_no_op_on_selection():
    s = _sampler()
    idx = list(range(100))
    scores = [float((7 * i) % 100) for i in idx]
    s.add_to_candidates(idx, scores)
    diag = s.refresh_scores(dict(zip(idx, scores)))
    assert diag["rescore/topk_overlap"] == 1.0
    assert diag["rescore/mean_stale"] == diag["rescore/mean_fresh"]


def test_refresh_only_touches_known_candidates_and_only_in_wake():
    s = _sampler()
    s.add_to_candidates([0, 1], [1.0, 2.0])
    diag = s.refresh_scores({0: 5.0, 999: 3.0})
    assert 999 not in s.wake_candidates and s.wake_candidates[0][0] == 5.0
    assert diag["rescore/n"] == 1
    s.switch_phase("SLEEP")
    with pytest.raises(AssertionError):
        s.refresh_scores({0: 1.0})


def test_fresh_config_differs_from_sh_rr_strict_only_in_rescore():
    conf = os.path.join(os.path.dirname(__file__), "..", "conf", "sleep_mechanism")
    with open(os.path.join(conf, "sh_rr_strict.yaml")) as f:
        base = yaml.safe_load(f)
    with open(os.path.join(conf, "sh_rr_strict_fresh.yaml")) as f:
        fresh = yaml.safe_load(f)
    assert fresh.pop("replay_rescore") is True
    assert fresh == base
