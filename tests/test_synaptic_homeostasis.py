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


# --- 3. per-tensor vs global thresholds under scale heterogeneity ---


def scale_heterogeneous_scores(model: nn.Module) -> dict:
    """Scores where one tensor's values are three orders of magnitude larger
    than every other tensor's. A single pooled quantile cannot see past it."""
    scores = {}
    for name, param in model.named_parameters():
        base = torch.linspace(0.0, 1.0, param.numel()).reshape(param.shape)
        scores[name] = base * (1000.0 if name == "lm_head.weight" else 1.0)
    return scores


def protected_fraction(model, scores, scope, protect=0.25):
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    apply_fisher_protected_shrink(
        fisher=scores,
        module=model,
        shrink_factor=SHRINK,
        protect_top_fraction=protect,
        threshold_scope=scope,
    )
    out = {}
    for name, param in model.named_parameters():
        untouched = torch.isclose(param.data, before[name])
        out[name] = untouched.float().mean().item()
    return out


def test_per_tensor_scope_protects_the_fraction_inside_every_tensor():
    model = ToyLM()
    fractions = protected_fraction(
        model, scale_heterogeneous_scores(model), "per_tensor"
    )
    for name, frac in fractions.items():
        assert 0.15 < frac < 0.35, (
            f"{name} protected {frac:.2%}, expected ~25% — per-tensor ranking "
            "should allocate the fraction within each tensor"
        )


def test_global_scope_lets_one_tensor_dominate_the_cutoff():
    """Documents why per_tensor is the default. With pooled ranking the
    large-magnitude tensor absorbs the whole protected budget and the others
    get essentially nothing."""
    model = ToyLM()
    fractions = protected_fraction(
        model, scale_heterogeneous_scores(model), "global"
    )
    assert fractions["lm_head.weight"] > 0.5
    assert fractions["dense.weight"] < 0.05


def test_unknown_threshold_scope_rejected():
    model = ToyLM()
    with pytest.raises(ValueError, match="threshold_scope"):
        apply_fisher_protected_shrink(
            fisher=uniform_scores(model),
            module=model,
            shrink_factor=SHRINK,
            protect_top_fraction=0.2,
            threshold_scope="per_layer",
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


# --- 4. sign semantics of the Taylor saliency ---


def taylor_scores(weights, grads, signal):
    """Mirror of CustomTrainer._finalize_taylor_saliency's scoring rule.

    Kept here as an executable statement of the convention: whatever the
    signal, a HIGHER stored score must mean "more worth protecting".
    """
    saliency = weights * grads
    if signal == "taylor_signed":
        return -saliency
    return saliency.abs()


def test_signed_taylor_protects_weights_whose_shrink_would_raise_loss():
    """Shrinking w by (1 - f) moves the loss by about -(1 - f) * w * grad.
    So a negative w*grad means shrinking hurts -> protect. A positive w*grad
    means shrinking helps -> shrink freely."""
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
    # products: -5 (shrink hurts most), -1, +1, +5 (shrink helps most)
    grads = torch.tensor([-5.0, -1.0, 1.0, 5.0])

    scores = taylor_scores(weights, grads, "taylor_signed")

    # Highest score must be the most-negative product.
    assert int(scores.argmax()) == 0
    # And protecting the top 25% must select exactly that weight.
    model = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(weights.reshape(1, 4))
    before = model.weight.detach().clone()
    apply_fisher_protected_shrink(
        fisher={"weight": scores.reshape(1, 4)},
        module=model,
        shrink_factor=SHRINK,
        protect_top_fraction=0.25,
        threshold_scope="per_tensor",
    )
    assert torch.isclose(
        model.weight[0, 0], before[0, 0]
    ), "the weight whose shrink would raise the loss was not protected"
    assert torch.isclose(model.weight[0, 3], before[0, 3] * SHRINK)


def test_abs_taylor_protects_largest_magnitude_product():
    """Molchanov's criterion ignores direction, so the +5 product ties with
    the -5 one and both outrank the small products."""
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
    grads = torch.tensor([-5.0, -1.0, 1.0, 5.0])

    scores = taylor_scores(weights, grads, "taylor_abs")

    assert torch.allclose(scores, torch.tensor([5.0, 1.0, 1.0, 5.0]))
    # The two conventions disagree: signed ranks index 3 last, abs ranks it
    # joint-first. This is exactly the arm the falsification plan compares.
    signed = taylor_scores(weights, grads, "taylor_signed")
    assert int(signed.argmin()) == 3
    assert int(scores.argmax()) in (0, 3)


def test_zero_gradient_weights_are_never_protected_by_either_signal():
    """A weight with no gradient has no first-order effect either way, so it
    must not outrank a weight that actually matters."""
    weights = torch.tensor([1.0, 1.0, 1.0])
    grads = torch.tensor([-4.0, 0.0, 4.0])

    signed = taylor_scores(weights, grads, "taylor_signed")
    absolute = taylor_scores(weights, grads, "taylor_abs")

    assert signed[1] < signed[0]
    assert absolute[1] < absolute[0]


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
