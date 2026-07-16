from .registry import register_model

"""
Model classes that use the GPT2 Architecture
"""

from transformers import GPT2Model as _GPT2Model
from transformers import GPT2Config

@register_model("gpt2", GPT2Config)
class GPT2Model(_GPT2Model):
    pass