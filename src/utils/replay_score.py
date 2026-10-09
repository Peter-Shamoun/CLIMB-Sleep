"""Per-sample replay score used by the loss criterion."""
from typing import Dict

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


@torch.no_grad()
def rescore_losses(
    model, dataset, indices, collator, task_name: str, device, batch_size: int = 512
) -> Dict[int, float]:
    """Loss-criterion score of every sample in ``indices`` under the model as it
    is now (eval mode, no dropout), batched through the wake collator.

    The wake phase records each sample's score once, at the step it is trained
    on, and the sleep sampler never updates it; this re-measures all of them
    so replay selection ranks candidates by their current loss (the
    stale-score test, paper_results.md Experiment 9). CLM only: an MLM score
    would depend on a fresh random mask and consume the training RNG.

    Args:
        dataset: indexable by a list of ints, returning a dict of columns
            (a HF ``datasets.Dataset``); needs ``input_ids``.
        collator: the wake collator (``DataCollatorForLanguageModeling`` with
            ``mlm=False``), which sets padded label positions to -100.
    Returns:
        {index: mean token cross-entropy}.
    """
    if task_name != "clm":
        raise ValueError(f"rescore_losses supports task 'clm' only, got {task_name!r}")
    was_training = model.training
    model.eval()
    order = sorted(int(i) for i in indices)  # sequential reads from the Arrow file
    scores: Dict[int, float] = {}
    try:
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            rows = dataset[chunk]
            batch = collator([{"input_ids": ids} for ids in rows["input_ids"]])
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
            logits = model(input_ids=input_ids, attention_mask=attention_mask)[0].transpose(-1, -2)
            labels = batch["labels"].to(device)
            per_sample = per_sample_mean_token_loss(
                logits[:, :, :-1].contiguous(), labels[:, 1:].contiguous()
            )
            scores.update(zip(chunk, per_sample.float().cpu().tolist()))
    finally:
        if was_training:
            model.train()
    return scores
