"""Multiplicative Synaptic Homeostasis (Plasticity Decay Approach 1).

After each sleep phase, scale down attention/FFN weights, protect high-magnitude
weights, and zero out sub-threshold weights. Ref: de Vivo et al. (2017) Science.
"""

import torch


def _is_target_param(name: str) -> bool:
    return name.endswith(".weight") and any(
        k in name for k in ("attention", "intermediate", "output")
    ) and "LayerNorm" not in name


def apply_synaptic_homeostasis(
    model: torch.nn.Module,
    decay_factor: float = 0.9,
    protect_percentile: float = 95.0,
    zero_threshold: float = 1e-4,
) -> dict:
    n_scaled = n_protected = n_zeroed = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not _is_target_param(name):
                continue
            mag = p.data.abs()
            threshold = torch.quantile(mag.float(), protect_percentile / 100.0)
            protect = mag > threshold
            saved = p.data[protect].clone()
            p.data.mul_(decay_factor)
            p.data[protect] = saved
            zero_mask = p.data.abs() < zero_threshold
            p.data[zero_mask] = 0.0
            n_scaled += p.numel()
            n_protected += protect.sum().item()
            n_zeroed += zero_mask.sum().item()
    return {"n_params_scaled": n_scaled, "n_params_protected": n_protected, "n_params_zeroed": n_zeroed}
