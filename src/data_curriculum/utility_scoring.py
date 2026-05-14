"""Need and Utility scoring helpers for Gain x Need replay selection.

The trainer calls ``compute_need_scores`` at the end of each wake phase to
produce a {sample_index: density_score} dict in eval mode. The sampler then
combines Need with Gain (per-sample squared gradient norms collected during
wake) via softmax-with-temperature in ``sample_utility_indices``.

Mattar & Daw (2018): utility = gain * need. Need is operationalized as k-NN
density in the model's embedding space - a sample sitting near many other
training samples is "representative" of the dataset.
"""

import logging
from typing import Dict, List

import faiss
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def compute_need_scores(
    model: Module,
    dataset: Dataset,
    candidate_indices: List[int],
    n_layers: int = 4,
    k: int = 50,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 32,
) -> Dict[int, float]:
    """k-NN density score per candidate sample.

    Forwards each candidate through ``model`` with ``output_hidden_states=True``,
    mean-pools the last ``n_layers`` hidden states across non-padding tokens,
    averages across layers, L2-normalizes, then queries FAISS ``IndexFlatIP``
    for the k nearest neighbors of each sample. Need = mean cosine similarity
    to those k neighbors (self excluded).
    """
    was_training = model.training
    model.eval()
    try:
        embeddings = _extract_embeddings(
            model, dataset, candidate_indices, n_layers, device, batch_size
        )
    finally:
        if was_training:
            model.train()

    embeddings_np = embeddings.numpy().astype("float32")
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    embeddings_np = embeddings_np / np.clip(norms, a_min=1e-12, a_max=None)

    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    # Query k+1 because the first hit is the sample itself (similarity ~1).
    n_query = min(k + 1, embeddings_np.shape[0])
    sims, _ = index.search(embeddings_np, n_query)
    # Drop self-column, average the remaining neighbors.
    if n_query > 1:
        need_scores = sims[:, 1:].mean(axis=1)
    else:
        # Degenerate case: only one candidate. Return zeros.
        need_scores = np.zeros(embeddings_np.shape[0], dtype=np.float32)

    return {
        int(idx): float(score)
        for idx, score in zip(candidate_indices, need_scores)
    }


def _extract_embeddings(
    model: Module,
    dataset: Dataset,
    indices: List[int],
    n_layers: int,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    """Mean-pool last ``n_layers`` hidden states across non-padding tokens."""
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            input_ids = torch.tensor(
                [dataset[int(i)]["input_ids"] for i in batch_idx],
                device=device,
            )
            attn_mask = torch.tensor(
                [dataset[int(i)]["attention_mask"] for i in batch_idx],
                device=device,
            )
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
            # outputs.hidden_states includes the embedding-layer output at [0]
            # and one tensor per transformer layer; restrict to transformer outputs.
            num_transformer_layers = len(outputs.hidden_states) - 1
            assert n_layers <= num_transformer_layers, (
                f"need_embedding_layers={n_layers} exceeds transformer layer "
                f"count {num_transformer_layers}"
            )
            hidden_states = outputs.hidden_states[-n_layers:]
            mask = attn_mask.unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp(min=1)
            pooled_per_layer = [
                (h * mask).sum(dim=1) / denom for h in hidden_states
            ]
            embedding = torch.stack(pooled_per_layer, dim=0).mean(dim=0)
            embeddings.append(embedding.detach().cpu())
    return torch.cat(embeddings, dim=0)


def sample_utility_indices(
    gain_scores: Dict[int, float],
    need_scores: Dict[int, float],
    num_replay: int,
    temperature_gain: float = 1.0,
    temperature_need: float = 1.0,
) -> List[int]:
    """Sample replay indices proportional to softmax(Gain/T_G) * softmax(Need/T_N).

    Both score dicts must have identical key sets (the wake-fold candidate pool).
    """
    if not gain_scores:
        raise ValueError("gain_scores is empty; cannot sample utility buffer")
    missing = set(gain_scores).symmetric_difference(need_scores)
    if missing:
        raise ValueError(
            f"gain and need keys disagree on {len(missing)} indices"
        )

    keys = list(gain_scores.keys())
    gain_t = torch.tensor([gain_scores[k] for k in keys], dtype=torch.float32)
    need_t = torch.tensor([need_scores[k] for k in keys], dtype=torch.float32)

    gain_probs = torch.softmax(gain_t / temperature_gain, dim=0)
    need_probs = torch.softmax(need_t / temperature_need, dim=0)
    utility = gain_probs * need_probs
    total = utility.sum()
    if total <= 0 or not torch.isfinite(total):
        logger.warning(
            "Utility distribution degenerate (sum=%s); falling back to uniform.",
            total,
        )
        utility = torch.ones_like(utility)
        total = utility.sum()
    utility = utility / total

    num_replay = min(num_replay, len(keys))
    # Sharp temperatures can leave fewer non-trivial entries than num_replay,
    # in which case multinomial-without-replacement samples zero-prob entries.
    nontrivial = int((utility > 0).sum().item())
    if nontrivial < num_replay:
        logger.warning(
            "Utility has %d non-zero entries but %d replay slots requested; "
            "temperatures may be too sharp.",
            nontrivial,
            num_replay,
        )
    sampled = torch.multinomial(
        utility, num_samples=num_replay, replacement=False
    )
    return [keys[i] for i in sampled.tolist()]
