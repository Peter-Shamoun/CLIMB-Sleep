import logging

from transformers import PreTrainedModel

# typing inmports
from ..config import BabyLMConfig
from .registry import CONFIG_REGISTRY, MODEL_REGISTRY
from .roberta import *
from .gpt2 import *

# A logger for this file
logger = logging.getLogger(__name__)


def load_base_model(cfg: BabyLMConfig) -> PreTrainedModel:
    """Loads the base model from the config file"""

    model_kwargs = cfg.model.model_kwargs

    # NOTE: The only required parameter is vocab_size and hidden_size
    # These two values effectively represent the input -> output dimensions of the model
    assert (
        "hidden_size" in model_kwargs and "vocab_size" in model_kwargs
    ), "hidden_size and vocab_size must at a minimum be specified in model_kwargs"

    if cfg.model.name in MODEL_REGISTRY:
        config = CONFIG_REGISTRY[cfg.model.name](**model_kwargs)
        # vmap (per-sample grads) doesn't work with SDPA attention — force eager.
        config._attn_implementation = "eager"

        if config.name_or_path:
            model = MODEL_REGISTRY[cfg.model.name].from_pretrained(
                config.name_or_path, config=config
            )
            logger.info(f"Loaded model config from {config.name_or_path}")
        else:
            logger.info(f"Initialized model config from scratch")
            model = MODEL_REGISTRY[cfg.model.name](config)
    else:
        raise ValueError(f"Model {cfg.model.name} not found in registry")

    # The final pooler layer is never used, gradients need to be deactivated
    for name, param in model.named_parameters():
        if "pooler" in name:
            param.requires_grad = False

    logger.debug("Model parameters:")
    for i, (name, param) in enumerate(model.named_parameters()):
        logger.debug(f"{i}: {name} - Requires grad: {param.requires_grad}")

    return model


def build_inference_lm(cfg: BabyLMConfig, trained_model: PreTrainedModel) -> PreTrainedModel:
    """Return a full LM (trunk + output head) carrying the trained weights.

    Used when exporting ``lm_model/`` for the BabyLM eval pipeline. When the
    training model already is the LM-head class (``gpt2_clm``,
    ``roberta_pre_layer_norm_mlm``) its whole state dict is copied, head
    included. Grafting only ``base_model`` into a fresh LM (the previous
    behaviour) left the freshly initialised head in place whenever
    ``tie_word_embeddings`` is false, so every exported model scored at
    chance on BLiMP regardless of training. For a trunk-only training model
    the trunk is grafted as before.
    """
    lm_model = load_base_model(cfg)
    if type(trained_model) is type(lm_model):
        lm_model.load_state_dict(trained_model.state_dict())
    else:
        setattr(lm_model, lm_model.base_model_prefix, trained_model.base_model)
    return lm_model
