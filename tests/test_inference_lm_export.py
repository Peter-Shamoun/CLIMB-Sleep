"""The exported lm_model/ must carry the trained LM head, not a fresh one.

Regression test for the near-chance BLiMP scores of every CLM run evaluated
from lm_model/ (Sep 2026): _initialize_full_lm_model grafted only base_model
into a freshly initialised GPT2LMHeadModel, so lm_head.weight stayed at its
init (std 0.02) while the trainer checkpoint's head was trained (std 0.33).
"""
import os
import sys

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import build_inference_lm, load_base_model  # noqa: E402

TINY = {
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "hidden_size": 16,
    "intermediate_size": 32,
    "vocab_size": 64,
    "eos_token_id": 4,
    "bos_token_id": 3,
    "pad_token_id": 1,
    "tie_word_embeddings": False,
}


def _cfg(name, **extra):
    kw = dict(TINY, **extra)
    return OmegaConf.create({"model": {"name": name, "model_kwargs": kw}})


def _head_params(model):
    return {k: v for k, v in model.state_dict().items() if k.startswith("lm_head")}


@pytest.mark.parametrize("name", ["gpt2_clm", "roberta_pre_layer_norm_mlm"])
def test_exported_lm_keeps_trained_head(name):
    torch.manual_seed(0)
    cfg = _cfg(name, **({"max_position_embeddings": 32} if "gpt2" in name else {}))
    trained = load_base_model(cfg)
    heads = _head_params(trained)
    assert heads, f"{name} has no lm_head parameters to protect"
    # Make the head unmistakably "trained".
    with torch.no_grad():
        for p in trained.parameters():
            p.add_(0.5)

    exported = build_inference_lm(cfg, trained)
    assert type(exported) is type(trained)
    for k, v in trained.state_dict().items():
        assert torch.equal(exported.state_dict()[k], v), f"{k} not carried over"


def test_exported_lm_grafts_trunk_when_training_model_is_trunk_only():
    torch.manual_seed(0)
    cfg_trunk = _cfg("gpt2", max_position_embeddings=32)
    cfg_lm = _cfg("gpt2_clm", max_position_embeddings=32)
    trunk = load_base_model(cfg_trunk)
    with torch.no_grad():
        for p in trunk.parameters():
            p.add_(0.5)
    exported = build_inference_lm(cfg_lm, trunk)
    assert exported.transformer is trunk
    assert hasattr(exported, "lm_head")
