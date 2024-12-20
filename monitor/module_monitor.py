from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from torch import nn
import math
from transformers import LlamaForCausalLM, LlamaModel
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb,LlamaDecoderLayer
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from typing import List, Optional, Tuple, Union
import time


