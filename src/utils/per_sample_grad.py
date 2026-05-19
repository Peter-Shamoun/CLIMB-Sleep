"""Per-sample gradient utilities backed by torch.func.vmap.

Shared by Gain x Need replay selection (Method 1) and online empirical Fisher
plasticity decay (Method 2). A single vmap call produces per-sample gradients;
both methods consume the same result.

Callers must pass the unwrapped base model (no DDP wrapper) and the MLM head
as separate nn.Modules. Inputs come from a normal MLM batch:
    input_ids:       [B, S]
    attention_mask:  [B, S]
    labels:          [B, S]  (-100 at non-masked positions, per the trainer)

The per-sample loss is the MLM loss averaged over the unmasked positions of
each sample, using a clamped denominator so a sample with zero unmasked
positions yields loss 0 (and therefore zero gradient) instead of NaN. This
matches the trainer's compute_loss formulation, so Gain = ||grad||^2 is a
first-order Taylor estimate of expected loss drop on a per-sample basis.
"""

from typing import Dict

import torch
from torch import Tensor
from torch.func import functional_call, grad, vmap
from torch.nn import Module
from torch.nn.functional import cross_entropy

def prepare_4d_mask(attention_mask, dtype):
    # [batch, seq_len] -> [batch, 1, 1, seq_len]
    mask = attention_mask[:, None, None, :].to(dtype)
    mask = (1.0 - mask) * torch.finfo(dtype).min
    return mask

def per_sample_grads(
    model: Module,
    mlm_head: Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
) -> Dict[str, Tensor]:
    """Per-sample gradients of the MLM loss w.r.t. trainable params.

    Returns a dict keyed by ``"model.<name>"`` or ``"mlm_head.<name>"``. Each
    value has shape ``[batch, *param_shape]``. Only ``requires_grad=True``
    parameters are included.
    """
    base_params = {
        k: v.detach() for k, v in model.named_parameters() if v.requires_grad
    }
    head_params = {
        k: v.detach()
        for k, v in mlm_head.named_parameters()
        if v.requires_grad
    }
    base_buffers = {k: v.detach() for k, v in model.named_buffers()}
    head_buffers = {k: v.detach() for k, v in mlm_head.named_buffers()}

    def loss_fn(base_p, head_p, base_b, head_b, x, attn, y):
        x_b = x.unsqueeze(0)
        attn_b = attn.unsqueeze(0)
        y_b = y.unsqueeze(0)
        base_out = functional_call(
            model,
            (base_p, base_b),
            args=(),
            kwargs={"input_ids": x_b, "attention_mask": attn_b},
        )
        hidden = base_out[0]
        logits = functional_call(
            mlm_head,
            (head_p, head_b),
            args=(hidden,),
        ).transpose(-1, -2)
        # Mean over unmasked positions with clamp(min=1). All-(-100) samples
        # would otherwise produce 0/0 = NaN here, propagating into the
        # per-sample gradient and contaminating Gain + Fisher.
        per_token = cross_entropy(logits, y_b, reduction="none")
        mask = (y_b != -100).float()
        return (per_token * mask).sum() / mask.sum().clamp(min=1)

    dtype = next(model.parameters()).dtype
    attention_mask_4d = prepare_4d_mask(attention_mask, dtype)
    
    grad_fn = grad(loss_fn, argnums=(0, 1))
    # randomness='different' so dropout (or any RNG op) gets an independent mask per
    # sample inside vmap, matching what a regular batched forward would produce.
    per_sample = vmap(
        grad_fn,
        in_dims=(None, None, None, None, 0, 0, 0),
        randomness="different",
    )
    # base_grads, head_grads = per_sample(
    #     base_params,
    #     head_params,
    #     base_buffers,
    #     head_buffers,
    #     input_ids,
    #     attention_mask_4d,
    #     labels,
    # )
    chunk_base_grads = []
    chunk_head_grads = []
    batch_size = input_ids.shape[0]
    chunk_size = 8 # TODO: make this a config parameter somehow

    for start in range(0, batch_size, chunk_size):
        bg, hg = per_sample(
            base_params, head_params, base_buffers, head_buffers,
            input_ids[start:start+chunk_size],
            attention_mask_4d[start:start+chunk_size],
            labels[start:start+chunk_size],
        )
        chunk_base_grads.append(bg)
        chunk_head_grads.append(hg)

    # bg and hg are dicts of {param_name: [chunk_size, *param_shape]}
    base_grads = {k: torch.cat([c[k] for c in chunk_base_grads], dim=0) for k in base_params}
    head_grads  = {k: torch.cat([c[k] for c in chunk_head_grads],  dim=0) for k in head_params}

    out: Dict[str, Tensor] = {}
    for k, v in base_grads.items():
        out[f"model.{k}"] = v
    for k, v in head_grads.items():
        out[f"mlm_head.{k}"] = v
    return out


def per_sample_squared_grad_norms(grads: Dict[str, Tensor]) -> Tensor:
    """Sum of squared gradient elements per sample. Returns shape ``[batch]``."""
    total = None
    for g in grads.values():
        flat = g.reshape(g.shape[0], -1)
        contrib = (flat * flat).sum(dim=1)
        total = contrib if total is None else total + contrib
    return total
