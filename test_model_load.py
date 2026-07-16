import logging
import os
import argparse
import math
import torch

import hydra
from hydra.core.config_store import ConfigStore

from src.config import BabyLMConfig

# type-checks dynamic config file
cs = ConfigStore.instance()
cs.store(name="base_config", node=BabyLMConfig)

# A logger for this file
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: BabyLMConfig):
    logger.info("Initializing model")
    model = load_base_model(cfg)

    print("=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    print(model.config)
    print()

    print("=" * 60)
    print("PARAMETER COUNTS")
    print("=" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print()

    print("=" * 60)
    print("MODULE STRUCTURE (top level)")
    print("=" * 60)
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"{name:20s} {n_params:>15,} params  {type(module).__name__}")
    print()

    print("=" * 60)
    print("WEIGHT INIT SANITY CHECK (mean/std per param group)")
    print("=" * 60)
    for name, p in model.named_parameters():
        print(f"{name:45s} shape={str(list(p.shape)):20s} "
            f"mean={p.data.mean().item(): .5f}  std={p.data.std().item(): .5f}")
    print()

    print("=" * 60)
    print("NaN / Inf CHECK")
    print("=" * 60)
    bad = [name for name, p in model.named_parameters()
        if torch.isnan(p).any() or torch.isinf(p).any()]
    print("Clean — no NaNs/Infs" if not bad else f"WARNING: bad values in {bad}")
    print()

    print("=" * 60)
    print("WEIGHT TYING CHECK (wte vs lm_head)")
    print("=" * 60)
    if hasattr(model, "lm_head"):
        tied = model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
        print(f"lm_head tied to wte embedding: {tied}")
    print()

    print("=" * 60)
    print("DEVICE / DTYPE")
    print("=" * 60)
    first_param = next(model.parameters())
    print(f"Device: {first_param.device}")
    print(f"Dtype:  {first_param.dtype}")
    print()

    print("=" * 60)
    print("FORWARD PASS SHAPE CHECK")
    print("=" * 60)
    dummy_input = torch.randint(0, model.config.vocab_size, (2, 16))  # batch=2, seq_len=16
    model.eval()
    with torch.no_grad():
        out = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Logits shape: {out.logits.shape}")
    expected = (2, 16, model.config.vocab_size)
    print(f"Expected:     {expected}")
    print("MATCH" if tuple(out.logits.shape) == expected else "MISMATCH")

if __name__ == "__main__":
    main()