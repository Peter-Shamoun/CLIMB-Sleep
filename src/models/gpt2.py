from .registry import register_model

"""
Model classes that use the GPT2 Architecture
"""

from transformers import GPT2Model as _GPT2Model
from transformers import GPT2LMHeadModel as _GPT2LMHeadModel
from transformers import GPT2Config

@register_model("gpt2", GPT2Config)
class GPT2Model(_GPT2Model):
    pass

@register_model("gpt2_clm", GPT2Config)
class GPT2LMHeadModel(_GPT2LMHeadModel):
    pass