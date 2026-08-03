"""Post-sleep multiplicative shrink with importance-based protection.

Implements Method 2 of the Gain x Need + plasticity decay plan: after each
sleep phase, every weight is multiplied by ``shrink_factor`` except the top
``protect_top_fraction`` by accumulated empirical Fisher diagonal (squared
per-sample gradients averaged across the most recent wake phase). The
importance signal is computed during wake; see ``CustomTrainer``.

Score keys are the *plain* names produced by ``model.named_parameters()``,
matching what ``per_sample_grads`` returns. Do not reintroduce a prefix on
either side: a prefix on one side only silently zeroes out the whole
mechanism (every lookup misses, nothing is shrunk), which is exactly the
failure this module now raises on.

Tied weights (same storage referenced under multiple parameter names) are
detected by ``param.data.data_ptr()`` and shrunk only once. We deduplicate in
a first pass before mutating any weights, because ``param.data = torch.where(...)``
replaces the underlying storage and invalidates ptr-based detection mid-loop.
"""

import logging
from typing import Dict

import torch
from torch import Tensor
from torch.nn import Module

logger = logging.getLogger(__name__)


def apply_fisher_protected_shrink(
    fisher: Dict[str, Tensor],
    module: Module,
    shrink_factor: float,
    protect_top_fraction: float,
    threshold_scope: str = "per_tensor",
) -> None:
    """In-place: scale all weights by ``shrink_factor`` except the top
    ``protect_top_fraction`` by importance value.

    Args:
        fisher: ``{param_name: tensor}`` keyed exactly as
            ``module.named_parameters()`` and as returned by
            ``per_sample_grads``. Tensor shape matches the underlying parameter.
        module: the (unwrapped) model whose parameters are shrunk.
        shrink_factor: scalar in (0, 1] applied to non-protected weights.
        protect_top_fraction: fraction of weights kept at full magnitude.
        threshold_scope: ``"per_tensor"`` or ``"global"``; see
            ``_fisher_threshold``.

    Raises:
        RuntimeError: if a non-empty ``fisher`` matches no parameter. That
            means the key conventions on the two sides have drifted apart and
            the shrink would silently be a no-op.
    """
    if not fisher:
        logger.warning("Importance dict is empty; skipping shrink.")
        return
    if not 0.0 <= protect_top_fraction <= 1.0:
        raise ValueError(
            f"protect_top_fraction must be in [0, 1], got {protect_top_fraction}"
        )

    threshold = _fisher_threshold(
        fisher, protect_top_fraction, threshold_scope
    )
    # First pass: dedupe by storage ptr. Tied weights (e.g., word embeddings
    # shared with the LM head) produce the same data_ptr() and must be shrunk
    # only once. We collect BEFORE any modification because torch.where below
    # replaces param.data's storage, which would invalidate ptr-based detection
    # of subsequent tied references.
    seen_storage_ptrs = set()
    to_shrink = []
    skipped_tied = 0
    skipped_no_fisher = 0
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if name not in fisher:
            skipped_no_fisher += 1
            continue
        storage_ptr = param.data.data_ptr()
        if storage_ptr in seen_storage_ptrs:
            skipped_tied += 1
            continue
        seen_storage_ptrs.add(storage_ptr)
        to_shrink.append((name, param))

    if not to_shrink:
        raise RuntimeError(
            "Importance dict has %d entries but matched no trainable parameter "
            "of the model; the shrink would be a silent no-op. This means the "
            "score keys and model.named_parameters() keys have diverged. "
            "Example score key: %r. Example param key: %r."
            % (
                len(fisher),
                next(iter(fisher)),
                next((n for n, _ in module.named_parameters()), None),
            )
        )

    protected_elements = 0
    total_elements = 0
    with torch.no_grad():
        for name, param in to_shrink:
            protect = fisher[name] >= threshold[name]
            param.data = torch.where(
                protect, param.data, param.data * shrink_factor
            )
            protected_elements += int(protect.sum())
            total_elements += protect.numel()
    logger.info(
        "Importance shrink applied: shrunk=%d params, protected=%d/%d weights "
        "(%.1f%%), scope=%s, tied-skipped=%d, no-score-skipped=%d, "
        "shrink_factor=%.3f, protect_top_fraction=%.3f",
        len(to_shrink),
        protected_elements,
        total_elements,
        100.0 * protected_elements / max(total_elements, 1),
        threshold_scope,
        skipped_tied,
        skipped_no_fisher,
        shrink_factor,
        protect_top_fraction,
    )


def _fisher_threshold(
    fisher: Dict[str, Tensor],
    protect_top_fraction: float,
    threshold_scope: str = "per_tensor",
) -> Dict[str, Tensor]:
    """Per-parameter cutoff at or above which weights are protected.

    Always returns one cutoff per key so the consumer never has to branch.

    ``"per_tensor"`` takes a separate quantile inside each parameter tensor.
    This is the safe default: importance magnitudes are not comparable across
    tensors, so a single pooled quantile makes "top 20%" mean "top 20% of
    whichever tensor happens to carry the largest raw values" rather than
    top 20% of each. It does mean protection is allocated per layer, which is
    a departure from SHY's global renormalization — hence the escape hatch.

    ``"global"`` reproduces the pooled behaviour: one quantile over every
    score, broadcast to all keys.
    """
    if threshold_scope not in ("per_tensor", "global"):
        raise ValueError(
            "threshold_scope must be 'per_tensor' or 'global', got "
            f"{threshold_scope!r}"
        )

    def cutoff(values: Tensor) -> Tensor:
        if protect_top_fraction <= 0.0:
            # Shrink everything; cutoff above max so nothing is protected.
            return values.max() + 1.0
        if protect_top_fraction >= 1.0:
            # Protect everything; cutoff below min so all are protected.
            return values.min() - 1.0
        return torch.quantile(values, 1.0 - protect_top_fraction)

    if threshold_scope == "global":
        flat = torch.cat([f.flatten().float() for f in fisher.values()])
        shared = cutoff(flat)
        return {name: shared for name in fisher}

    return {
        name: cutoff(tensor.flatten().float())
        for name, tensor in fisher.items()
    }
