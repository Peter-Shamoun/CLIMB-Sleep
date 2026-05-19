"""Post-sleep multiplicative shrink with Fisher-based protection.

Implements Method 2 of the Gain x Need + plasticity decay plan: after each
sleep phase, every weight is multiplied by ``shrink_factor`` except the top
``protect_top_fraction`` by accumulated empirical Fisher diagonal (squared
per-sample gradients averaged across the most recent wake phase). The
importance signal is computed online during wake; see ``CustomTrainer``.

Tied weights (same storage referenced under multiple parameter names) are
detected by ``param.data.data_ptr()`` and shrunk only once. We deduplicate in
a first pass before mutating any weights, because ``param.data = torch.where(...)``
replaces the underlying storage and invalidates ptr-based detection mid-loop.
"""

import logging
from typing import Dict, Mapping

import torch
from torch import Tensor
from torch.nn import Module

logger = logging.getLogger(__name__)


def apply_fisher_protected_shrink(
    fisher: Dict[str, Tensor],
    modules: Mapping[str, Module],
    shrink_factor: float,
    protect_top_fraction: float,
) -> None:
    """In-place: scale all weights by ``shrink_factor`` except the top
    ``protect_top_fraction`` by Fisher value.

    Args:
        fisher: ``{"<prefix>.<param_name>": tensor}`` matching the layout
            returned by ``per_sample_grads`` (e.g., ``"model.X"``,
            ``"mlm_head.X"``). Tensor shape matches the underlying parameter.
        modules: ``{prefix: nn.Module}`` whose ``named_parameters()`` produce
            the suffix half of each fisher key.
        shrink_factor: scalar in (0, 1] applied to non-protected weights.
        protect_top_fraction: fraction of weights kept at full magnitude.
    """
    if not fisher:
        logger.warning("Fisher dict is empty; skipping shrink.")
        return
    if not 0.0 <= protect_top_fraction <= 1.0:
        raise ValueError(
            f"protect_top_fraction must be in [0, 1], got {protect_top_fraction}"
        )

    threshold = _fisher_threshold(fisher, protect_top_fraction)
    # First pass: dedupe by storage ptr. Tied weights (e.g., word_embeddings
    # shared with lm_head decoder) produce the same data_ptr() and must be
    # shrunk only once. We collect BEFORE any modification because torch.where
    # below replaces param.data's storage, which would invalidate ptr-based
    # detection of subsequent tied references.
    seen_storage_ptrs = set()
    to_shrink = []
    skipped_tied = 0
    skipped_no_fisher = 0
    for prefix, module in modules.items():
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            key = f"{prefix}.{name}"
            if key not in fisher:
                skipped_no_fisher += 1
                continue
            storage_ptr = param.data.data_ptr()
            if storage_ptr in seen_storage_ptrs:
                skipped_tied += 1
                continue
            seen_storage_ptrs.add(storage_ptr)
            to_shrink.append((key, param))

    with torch.no_grad():
        for key, param in to_shrink:
            protect = fisher[key] >= threshold
            param.data = torch.where(
                protect, param.data, param.data * shrink_factor
            )
    shrunk_params = len(to_shrink)
    logger.info(
        "Fisher shrink applied: shrunk=%d params, threshold=%.4e, "
        "tied-skipped=%d, no-fisher-skipped=%d, shrink_factor=%.3f, "
        "protect_top_fraction=%.3f",
        shrunk_params,
        float(threshold),
        skipped_tied,
        skipped_no_fisher,
        shrink_factor,
        protect_top_fraction,
    )


def _fisher_threshold(
    fisher: Dict[str, Tensor], protect_top_fraction: float
) -> Tensor:
    """Quantile cutoff above which weights are protected from shrink."""
    flat = torch.cat([f.flatten().float() for f in fisher.values()])
    if protect_top_fraction <= 0.0:
        # Shrink everything; threshold above max so nothing is protected.
        return flat.max() + 1.0
    if protect_top_fraction >= 1.0:
        # Protect everything; threshold below min so all are protected.
        return flat.min() - 1.0
    return torch.quantile(flat, 1.0 - protect_top_fraction)
