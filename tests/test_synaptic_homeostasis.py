"""CPU-only tests for the post-sleep shrink and the per-sample gradient contract.

These deliberately use a toy nn.Module rather than a real HF model: the shrink
code imports nothing but torch, and per_sample_grads calls functional_call on
whatever module it is handed. That keeps the whole suite runnable on a laptop
with no GPU and no transformers model load.

Run: python -m pytest tests/ -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn

from src.synaptic_homeostasis import apply_fisher_protected_shrink
from src.utils.per_sample_grad import per_sample_grads

SHRINK = 0.95


class ToyLM(nn.Module):
    """Minimal LM with the interface per_sample_grads expects.

    forward() takes input_ids + attention_mask and returns a tuple whose first
    element is [batch, seq, vocab] logits, matching HF model output indexing.
    """

    def __init__(self, vocab: int = 16, hidden: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.dense = nn.Linear(hidden, hidden)
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None):
        hidden = torch.tanh(self.dense(self.embed(input_ids)))
        return (self.lm_head(hidden),)


def uniform_scores(model: nn.Module) -> dict:
    """Importance dict keyed exactly like model.named_parameters()."""
    return {
        name: torch.ones_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }


# --- 1. key-match regression: the bug that made the shrink a silent no-op ---


def test_scores_keyed_like_named_parameters_reach_every_param():
    model = ToyLM()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    apply_fisher_protected_shrink(
        fisher=uniform_scores(model),
        module=model,
        shrink_factor=SHRINK,
        protect_top_fraction=0.0,
    )

    for name, param in model.named_parameters():
        assert torch.allclose(
            param.data, before[name] * SHRINK
        ), f"{name} was not shrunk; scores did not reach it"


def test_prefixed_keys_raise_instead_of_silently_doing_nothing():
    """A prefix on the score side only is the exact production bug. It must be
    loud: with prefixed keys nothing matches and the shrink would be a no-op.
    """
    model = ToyLM()
    prefixed = {f"model.{n}": t for n, t in uniform_scores(model).items()}

    with pytest.raises(RuntimeError, match="matched no trainable parameter"):
        apply_fisher_protected_shrink(
            fisher=prefixed,
            module=model,
            shrink_factor=SHRINK,
            protect_top_fraction=0.2,
        )


def test_empty_score_dict_is_a_warning_not_an_error():
    model = ToyLM()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    apply_fisher_protected_shrink(
        fisher={}, module=model, shrink_factor=SHRINK, protect_top_fraction=0.2
    )

    for name, param in model.named_parameters():
        assert torch.equal(param.data, before[name])


# --- 2. shrink arithmetic at the boundaries of protect_top_fraction ---


def test_protect_none_shrinks_everything():
    model = ToyLM()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    apply_fisher_protected_shrink(
        fisher=uniform_scores(model),
        module=model,
        shrink_factor=SHRINK,
        protect_top_fraction=0.0,
    )

    for name, param in model.named_parameters():
        assert torch.allclose(param.data, before[name] * SHRINK)


def test_protect_all_leaves_every_weight_untouched():
    model = ToyLM()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    apply_fisher_protected_shrink(
        fisher=uniform_scores(model),
        module=model,
        shrink_factor=SHRINK,
        protect_top_fraction=1.0,
    )

    for name, param in model.named_parameters():
        assert torch.allclose(param.data, before[name])


def test_protects_the_high_scoring_half():
    model = ToyLM()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    # Score the first row of dense.weight high, everything else low.
    scores = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    scores["dense.weight"][0, :] = 10.0

    apply_fisher_protected_shrink(
        fisher=scores,
        module=model,
        shrink_factor=SHRINK,
        protect_top_fraction=0.01,
    )

    dense = dict(model.named_parameters())["dense.weight"]
    assert torch.allclose(dense.data[0, :], before["dense.weight"][0, :])
    assert torch.allclose(
        dense.data[1, :], before["dense.weight"][1, :] * SHRINK
    )


def test_out_of_range_protect_fraction_rejected():
    model = ToyLM()
    with pytest.raises(ValueError, match="protect_top_fraction"):
        apply_fisher_protected_shrink(
            fisher=uniform_scores(model),
            module=model,
            shrink_factor=SHRINK,
            protect_top_fraction=1.5,
        )


# --- 5. tied weights are shrunk once, not twice ---


def test_aliased_storage_is_shrunk_once():
    """Two distinct Parameters sharing one storage (the alias case the
    data_ptr dedup exists for). The shared values must never take the shrink
    factor twice."""
    model = ToyLM()
    # Alias lm_head.weight's storage onto embed.weight's.
    model.lm_head.weight.data = model.embed.weight.data
    assert (
        model.lm_head.weight.data.data_ptr()
        == model.embed.weight.data.data_ptr()
    )
    before = model.embed.weight.detach().clone()

    apply_fisher_protected_shrink(
        fisher=uniform_scores(model),
        module=model,
        shrink_factor=SHRINK,
        protect_top_fraction=0.0,
    )

    for name in ("embed.weight", "lm_head.weight"):
        param = dict(model.named_parameters())[name]
        assert not torch.allclose(
            param.data, before * SHRINK * SHRINK
        ), f"{name} was shrunk twice through an aliased storage"


# --- 6. per_sample_grads key and shape contract ---


@pytest.mark.parametrize("task", ["mlm", "clm"])
def test_per_sample_grads_keys_match_named_parameters(task):
    torch.manual_seed(0)
    model = ToyLM()
    batch, seq = 3, 6
    input_ids = torch.randint(0, 16, (batch, seq))
    attention_mask = torch.ones(batch, seq)
    labels = torch.randint(0, 16, (batch, seq))
    if task == "clm":
        # The trainer shifts labels before calling; per_sample_grads only
        # trims the logits.
        labels = labels[:, 1:].contiguous()

    grads = per_sample_grads(model, input_ids, attention_mask, labels, task)

    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(grads.keys()) == expected, (
        "grad keys must match named_parameters exactly — this is the contract "
        "the shrink lookup depends on"
    )
    for name, grad in grads.items():
        param = dict(model.named_parameters())[name]
        assert grad.shape == (batch, *param.shape)
        assert torch.isfinite(grad).all()


def test_per_sample_grads_are_actually_per_sample():
    """Different samples must produce different gradients; a broadcast bug
    would make them identical and silently destroy the importance signal."""
    torch.manual_seed(0)
    model = ToyLM()
    input_ids = torch.tensor([[1, 2, 3, 4], [9, 9, 9, 9]])
    attention_mask = torch.ones(2, 4)
    labels = torch.tensor([[1, 2, 3, 4], [9, 9, 9, 9]])

    grads = per_sample_grads(model, input_ids, attention_mask, labels, "mlm")

    dense = grads["dense.weight"]
    assert not torch.allclose(dense[0], dense[1])


def test_ignore_index_labels_do_not_produce_nan():
    """An all-(-100) sample would divide by zero without the clamp in the
    loss; that NaN would poison the accumulator for the rest of the phase."""
    torch.manual_seed(0)
    model = ToyLM()
    input_ids = torch.randint(0, 16, (2, 4))
    attention_mask = torch.ones(2, 4)
    labels = torch.full((2, 4), -100)

    grads = per_sample_grads(model, input_ids, attention_mask, labels, "mlm")

    for name, grad in grads.items():
        assert torch.isfinite(grad).all(), f"{name} has non-finite grads"
