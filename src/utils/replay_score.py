"""Per-sample replay score used by the loss criterion."""
import torch
from torch.nn.functional import cross_entropy


def per_sample_mean_token_loss(
    logits: torch.Tensor, labels: torch.Tensor, **loss_kwargs
) -> torch.Tensor:
    """Mean cross-entropy over the scored tokens of each sample.

    Args:
        logits: (batch, vocab, seq) as produced by the trainer (already shifted
            for CLM: position t predicts labels[:, t]).
        labels: (batch, seq); positions equal to -100 (padding, unmasked MLM
            tokens) are excluded from both the sum and the count.
    Returns:
        (batch,) tensor; a sample with no scored token gets 0.
    """
    token_loss = cross_entropy(logits, labels, reduction="none", **loss_kwargs)
    mask = (labels != -100).float()
    return (token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
