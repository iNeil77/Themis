# Invoked as:
# CUDA_VISIBLE_DEVICES=0 python coderewardbench-seqcls.py "CodeShield/LibraRM-Inst-1.5B" --output "LibraRM-Inst-Aspect" --use-system-prompts --use-aspect-prompts --batch-size 32 --max-length 4096

"""
This script evaluates a scalar reward model on the Code RewardBench dataset,
calculating accuracy across different aspects and programming languages.
"""

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel
)
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import argparse
import json
import os


from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

INTERNLM2_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


# Modified from transformers.model.llama.configuration_llama.LlamaConfig
class InternLM2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternLM2Model`]. It is used to instantiate
    an InternLM2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the InternLM2-7B.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the InternLM2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`InternLM2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. InternLM2 supports up to 32768 tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism)
            to understand more about it. This value is necessary to ensure exact reproducibility
            of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        reward_token_id (`int`, *optional*, defaults to 92527):
            Token id used to calculate the reward score.
    """
    _auto_class = "AutoConfig"
    model_type = "internlm2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(  # pylint: disable=W0102
        self,
        vocab_size=103168,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        bias=True,
        rope_theta=10000,
        rope_scaling=None,
        attn_implementation=None,
        reward_token_id=92527,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.bias = bias

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attn_implementation = attn_implementation
        if self.attn_implementation is None:
            self.attn_implementation = "eager"
        self.reward_token_id = reward_token_id

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if (
            rope_scaling_factor is None
            or not isinstance(rope_scaling_factor, (float, int))
            or rope_scaling_factor < 1.0
        ):
            raise ValueError(
                f"`rope_scaling`'s factor field must be a number >= 1, got {rope_scaling_factor} "
                f"of type {type(rope_scaling_factor)}"
            )


"""PyTorch InternLM2 model."""
import math
import queue
import threading
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

try:
    from transformers.generation.streamers import BaseStreamer
except Exception:
    BaseStreamer = None


try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except:
    pass


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "InternLM2Config"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))  # pylint: disable=E1102
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class InternLM2RMSNorm(nn.Module):
    """InternLM2RMSNorm is equivalent to T5LayerNorm."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(InternLM2RMSNorm)


class InternLM2RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for the InternLM2 model. Credits to the Reddit user /u/lucidrains."""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class InternLM2LinearScalingRotaryEmbedding(InternLM2RotaryEmbedding):
    """InternLM2RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin


class InternLM2DynamicNTKScalingRotaryEmbedding(InternLM2RotaryEmbedding):
    """InternLM2RotaryEmbedding extended with Dynamic NTK scaling.
    Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):  # pylint: disable=unused-argument
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class InternLM2MLP(nn.Module):
    """MLP for InternLM2 model."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w3 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.w2(self.act_fn(self.w1(x)) * self.w3(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class InternLM2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternLM2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.wqkv = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.bias,
        )
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)

        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = InternLM2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = InternLM2LinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = InternLM2DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,  # pylint: disable=unused-argument
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            # split qkv_states by tp size
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            qkv_slices = self.wqkv.weight.split(key_value_slicing, dim=0)
            qkv_states = torch.cat(
                [F.linear(hidden_states, qkv_slice) for qkv_slice in qkv_slices], dim=-1  # pylint: disable=E1102
            )
        else:
            qkv_states = self.wqkv(hidden_states)

        qkv_states = rearrange(
            qkv_states,
            "b q (h gs d) -> b q h gs d",
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )

        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d").transpose(1, 2)
        key_states = qkv_states[..., -2, :].transpose(1, 2)
        value_states = qkv_states[..., -1, :].transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.wo.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])  # pylint: disable=E1102
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class InternLM2FlashAttention2(InternLM2Attention):
    """
    InternLM2 flash attention module. This module inherits from `InternLM2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement,
        #   that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        # Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1)
        #   produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at "
                "https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.wqkv(hidden_states)

        qkv_states = rearrange(
            qkv_states,
            "b q (h gs d) -> b q h gs d",
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )

        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout
        # [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # dropout_rate = self.attention_dropout if self.training else 0.0
        dropout_rate = 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (InternLM2RMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.wqkv.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value  # pylint: disable=E0606

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.
        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1.
            # For details, please see the comment in InternLM2FlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(  # pylint: disable=E0606
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)  # pylint: disable=E0606
        else:
            attn_output = flash_attn_func(  # pylint: disable=E0606
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(  # pylint: disable=E0606
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(  # pylint: disable=E0606
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(  # pylint: disable=E0606
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(  # pylint: disable=E0606
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


# Copied from transformers.models.llama.modeling_llama.LllamaSdpaAttention with Llama->InternLM2
class InternLM2SdpaAttention(InternLM2Attention):
    """
    InternLM2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `InternLM2Attention` as the weights of the module stays untouched. The only changes are on the forward pass
    to adapt to SDPA API.
    """

    # Adapted from InternLM2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"`
            # once this is implemented.
            logger.warning_once(
                "InternLM2Model uses InternLM2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` "
                "does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                "but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. "
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.wqkv(hidden_states)

        qkv_states = rearrange(
            qkv_states,
            "b q (h gs d) -> b q h gs d",
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )

        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with
        # custom attn_mask, Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of
        # an inline conditional assignment in SDPA to support both torch.compile's dynamic shapes and full graph
        # options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = bool(causal_mask is None and q_len > 1)

        attn_output = torch.nn.functional.scaled_dot_product_attention(  # pylint: disable=E1102
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.wo(attn_output)

        return attn_output, None, past_key_value


INTERNLM2_ATTENTION_CLASSES = {
    "eager": InternLM2Attention,
    "flash_attention_2": InternLM2FlashAttention2,
    "sdpa": InternLM2SdpaAttention,
}


# Modified from transformers.models.llama.modeling_llama.LlamaDecoderLayer with Llama->InternLM2
class InternLM2DecoderLayer(nn.Module):
    """InternLM2 Decoder Layer. This module is a single layer of the InternLM2 model."""

    def __init__(self, config: InternLM2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.attention = INTERNLM2_ATTENTION_CLASSES[config.attn_implementation](config=config, layer_idx=layer_idx)

        self.feed_forward = InternLM2MLP(config)
        self.attention_norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


InternLM2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`InternLM2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


# Copied from transformers.models.llama.modeling_llama.LlamaPreTrainedModel with Llama->InternLM2
@add_start_docstrings(
    "The bare InternLM2 Model outputting raw hidden-states without any specific head on top.",
    InternLM2_START_DOCSTRING,
)
class InternLM2PreTrainedModel(PreTrainedModel):
    """
    InternLM2 pretraiend model's base class.
    """

    config_class = InternLM2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["InternLM2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


InternLM2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).
            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.
            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.
            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.
            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.
            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


# Modified from transformers.models.llama.modeling_llama.LlamaModel with Llama->InternLM2
@add_start_docstrings(
    "The bare InternLM2 Model outputting raw hidden-states without any specific head on top.",
    InternLM2_START_DOCSTRING,
)
class InternLM2Model(InternLM2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`InternLM2DecoderLayer`]
    Args:
        config: InternLM2Config
    """

    _auto_class = "AutoModel"

    def __init__(self, config: InternLM2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList(
            [InternLM2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.tok_embeddings

    def set_input_embeddings(self, value):
        self.tok_embeddings = value

    @add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length
        # even when the static KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at
        # each decode steps due to the dynamic shapes. (`recording cudagraph tree for symint key 13`, etc.), which is
        # VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using `fullgraph=True`.
        # See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config.attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config.attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config.attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)  # pylint: disable=E1120

        return causal_mask


# Modified from transformers.models.llama.modeling_llama.LlamaForCausalLM
class InternLM2ForCausalLM(InternLM2PreTrainedModel):
    """Causal language model (CLM) for InternLM2."""

    _auto_class = "AutoModelForCausalLM"
    _tied_weights_keys = ["output.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = InternLM2Model(config)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        return self.output

    def set_output_embeddings(self, new_embeddings):
        self.output = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, InternLM2ForCausalLM
        >>> model = InternLM2ForCausalLM.from_pretrained("meta-InternLM2/InternLM2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-InternLM2/InternLM2-2-7b-hf")
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            output_slices = self.output.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [
                F.linear(hidden_states, output_slices[i])  # pylint: disable=not-callable
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.output(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_cache_shape(), device=input_ids.device)
                    if past_key_values.get_max_cache_shape() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]  # pylint: disable=E1130

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, meta_instruction=""):
        if history is None:
            history = []
        if tokenizer.add_bos_token:
            prompt = ""
        else:
            prompt = tokenizer.bos_token
        if meta_instruction:
            prompt += f"""<|im_start|>system\n{meta_instruction}<|im_end|>\n"""
        for record in history:
            prompt += f"""<|im_start|>user\n{record[0]}<|im_end|>\n<|im_start|>assistant\n{record[1]}<|im_end|>\n"""
        prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""
        return tokenizer([prompt], return_tensors="pt")

    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        streamer: Optional[BaseStreamer] = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.8,
        meta_instruction: str = "You are an AI assistant whose name is InternLM (书生·浦语).\n"
        "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory "
        "(上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such "
        "as English and 中文.",
        **kwargs,
    ):
        if history is None:
            history = []
        inputs = self.build_inputs(tokenizer, query, history, meta_instruction)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        # also add end-of-assistant token in eos token id to avoid unnecessary generation
        eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]
        outputs = self.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        outputs = outputs[0].cpu().tolist()[len(inputs["input_ids"][0]) :]
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        response = response.split("<|im_end|>")[0]
        history = history + [(query, response)]
        return response, history

    @torch.no_grad()
    def stream_chat(
        self,
        tokenizer,
        query: str,
        history: List[Tuple[str, str]] = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.8,
        **kwargs,
    ):
        if history is None:
            history = []
        """
        Return a generator in format: (response, history)
        Eg.
        ('你好，有什么可以帮助您的吗', [('你好', '你好，有什么可以帮助您的吗')])
        ('你好，有什么可以帮助您的吗？', [('你好', '你好，有什么可以帮助您的吗？')])
        """
        if BaseStreamer is None:
            raise ModuleNotFoundError(
                "The version of `transformers` is too low. Please make sure "
                "that you have installed `transformers>=4.28.0`."
            )

        response_queue = queue.Queue(maxsize=20)

        class ChatStreamer(BaseStreamer):
            """
            Streamer used in generate to print words one by one.
            """

            def __init__(self, tokenizer) -> None:
                super().__init__()
                self.tokenizer = tokenizer
                self.queue = response_queue
                self.query = query
                self.history = history
                self.response = ""
                self.cache = []
                self.received_inputs = False
                self.queue.put((self.response, history + [(self.query, self.response)]))

            def put(self, value):
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError("ChatStreamer only supports batch size 1")
                elif len(value.shape) > 1:
                    value = value[0]

                if not self.received_inputs:
                    # The first received value is input_ids, ignore here
                    self.received_inputs = True
                    return

                self.cache.extend(value.tolist())
                token = self.tokenizer.decode(self.cache, skip_special_tokens=True)
                if token.strip() != "<|im_end|>":
                    self.response = self.response + token
                    history = self.history + [(self.query, self.response)]
                    self.queue.put((self.response, history))
                    self.cache = []
                else:
                    self.end()

            def end(self):
                self.queue.put(None)

        def stream_producer():
            return self.chat(
                tokenizer=tokenizer,
                query=query,
                streamer=ChatStreamer(tokenizer=tokenizer),
                history=history,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

        def consumer():
            producer = threading.Thread(target=stream_producer)
            producer.start()
            while True:
                res = response_queue.get()
                if res is None:
                    return
                yield res

        return consumer()


# Copied from transformers.models.llama.modeling_llama.LlamaForSequenceClassification with Llama->InternLM2
@add_start_docstrings(
    """
    The InternLM2 Model transformer with a sequence classification head on top (linear layer).
    [`InternLM2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.
    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    InternLM2_START_DOCSTRING,
)
class InternLM2ForSequenceClassification(InternLM2PreTrainedModel):
    """Sequence Classification Head for InternLM2 Model."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = InternLM2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    @add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype in (torch.long, torch.int)):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


# Copied from transformers.models.llama.modeling_llama.LlamaForQuestionAnswering with Llama->InternLM2
@add_start_docstrings(
    """
The InternLM2 Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    InternLM2_START_DOCSTRING,
)
class InternLM2ForQuestionAnswering(InternLM2PreTrainedModel):
    """Question Answering model for InternLM2."""

    base_model_prefix = "transformer"

    def __init__(self, config):
        super().__init__(config)
        self.transformer = InternLM2Model(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.tok_embeddings

    def set_input_embeddings(self, value):
        self.transformer.tok_embeddings = value

    @add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Copied from transformers.models.llama.modeling_llama.LlamaForTokenClassification with Llama->InternLM2
@add_start_docstrings(
    """
    The InternLM2 Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    InternLM2_START_DOCSTRING,
)
class InternLM2ForTokenClassification(InternLM2PreTrainedModel):
    """Token classification model for InternLM2."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = InternLM2Model(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    @add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Modified from transformers.models.llama.modeling_llama.LlamaForTokenClassification
class InternLM2ForRewardModel(InternLM2PreTrainedModel):
    """InternLM2 backbone with a linear value head for scalar reward scoring."""

    _auto_class = "AutoModel"
    _tied_weights_keys = ["v_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = InternLM2Model(config)
        self.vocab_size = config.vocab_size
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.reward_token_id = config.reward_token_id

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        return self.v_head

    def set_output_embeddings(self, new_embeddings):
        self.v_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
        
        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.v_head(hidden_states)
        # get end reward token's score
        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1,1)

        # move tensor if on different device so auto_map works
        if ends.device != hidden_states.device:
            ends = ends.to(hidden_states.device)

        reward_scores = torch.gather(hidden_states.squeeze(-1), 1, ends)

        loss = None

        if not return_dict:
            output = (reward_scores,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=reward_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def get_score(
        self,
        tokenizer,
        conversation: List[dict],
        **kwargs,
    ):
        """
        Computes the reward score for a given conversation.
        This function takes a conversation represented as a list of dictionaries, formats it into a string using the chat 
        template from the tokenizer, and passes it through the model to compute the score. A special token representing 
        the reward score is appended to the input sequence. The reward score is then extracted from the model's output.
        Args:
            tokenizer: The tokenizer to be used for formatting and tokenizing the conversation.
            conversation (List[dict]): A list of dictionaries where each dictionary represents a message in the conversation.
        Returns:
            float: The computed reward score from the model.
        """
        conversation_str = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        input_ids = tokenizer.encode(conversation_str, return_tensors="pt", add_special_tokens=False)
        # add reward score token at the end of the input_ids if it is not already there
        if input_ids[0, -1] != self.reward_token_id:
            input_ids = torch.cat([input_ids, torch.tensor([[self.reward_token_id]], dtype=torch.long)], dim=1)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        outputs = self.forward(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device), **kwargs)
        score = outputs[0].detach().cpu().item()
        return score
    
    @torch.no_grad()
    def get_scores(
        self,
        tokenizer,
        conversations: List[List[dict]],
        **kwargs,
    ):
        """
        Computes the reward scores for multiple conversations in a batched manner.
        This function takes multiple conversations, each represented as a list of dictionaries, formats them into strings using the chat 
        template from the tokenizer, and passes these formatted strings through the model to compute scores for each conversation. 
        Each input sequence has a special token representing the reward score appended before passing to the model. 
        The reward scores are then extracted from the model's output.
        Args:
            tokenizer: The tokenizer to be used for formatting and tokenizing  the conversation.
            conversations (List[List[dict]]): A list of conversations, with each conversation represented as a list of dictionaries where each dictionary contains a message.
        Returns:
            List[float]: A list of computed reward scores for each conversation in the input batch.
        """
        conversation_strs = [tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False) for conversation in conversations]
        batch_input_ids = []
        attention_masks = []

        for conversation_str in conversation_strs:
            input_ids = tokenizer.encode(conversation_str, return_tensors="pt", add_special_tokens=False)
            # add reward score token at the end of the input_ids if it is not already there
            if input_ids[0, -1] != self.reward_token_id:
                input_ids = torch.cat([input_ids, torch.tensor([[self.reward_token_id]], dtype=torch.long)], dim=1)
            input_ids = input_ids.squeeze(0)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.bool)
            batch_input_ids.append(input_ids)
            attention_masks.append(attention_mask)

        r_pad_batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        r_pad_attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=False)

        outputs = self.forward(input_ids=r_pad_batch_input_ids.to(self.device), attention_mask=r_pad_attention_masks.to(self.device), **kwargs)
        scores = outputs[0].squeeze().detach().cpu().tolist()
        return scores
    
    @torch.no_grad()
    def compare(
        self,
        tokenizer,
        conversation1: List[dict],
        conversation2: List[dict],
        return_logits: bool = False,
        **kwargs,
    ):
        """
        Compares the reward scores of two conversations and determines which conversation has a higher score.
        This function computes reward scores for two given conversations using the `get_score` method and compares the scores to determine which conversation has a higher score.
        The function can optionally return the actual scores (logits) along with the comparison result.
        Parameters:
            tokenizer: The tokenizer used for formatting and tokenizing the conversation.
            conversation1 (List[dict]): The first conversation to compare, represented as a list of dictionaries where each dictionary contains a message.
            conversation2 (List[dict]): The second conversation to compare, similarly represented.
            return_logits (bool, optional): If True, the function returns both the comparison result and the actual scores of the two conversations. Defaults to False.
        Returns:
            bool: True if the score of the first conversation is greater than the second, otherwise False.
            List[float] (optional): A list containing the scores of the first and second conversations respectively.
        Note:
        - This function is designed for inference, with `@torch.no_grad()` used to disable gradient calculations to optimize performance.
        """
        score1 = self.get_score(tokenizer, conversation1, **kwargs)
        score2 = self.get_score(tokenizer, conversation2, **kwargs)
        if return_logits:
            return score1 > score2, [score1, score2]
        else:
            return score1 > score2
        
    @torch.no_grad()
    def rank(
        self,
        tokenizer,
        conversations: List[List[dict]],
        return_logits: bool = False,
        **kwargs,
    ):
        """
        Ranks the conversations based on their scores.
        Args:
            tokenizer: The tokenizer to be used for formatting and tokenizing  the conversation.
            conversations: A list of conversations, where each conversation is represented as a list of dictionaries. Each dictionary contains the necessary information for the conversation.
            return_logits: If True, returns the conversation indices along with their logits. Defaults to False.
        Returns:
            list: A list of conversation rank indices based on their scores. Smaller index means higher score.
            List[float] (optional): If return_logits is True, a list of conversation indices and their corresponding logits.
        """
        scores = self.get_scores(tokenizer, conversations, **kwargs)
        if return_logits:
            return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True), scores
        else:
            return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)



# System prompts for different coding aspects
SYSTEM_PROMPT_MAP = {

    'Full': """You are an experienced AI judge and evaluator, specifically designed and trained to assess the overall quality of responses to human prompts. The prompts can be of various types, including but not limited to open-ended questions, coding tasks, creative writing, requests for information, and more. The responses may be generated by a language model or a human and may include text, code, or both. You will receive the prompt and the generated response for each judgement session. Your exact mandate is to score the response based on a pre-specified but flexible set of criteria. 

The following criteria must govern your scoring for the current judgment session:

1. **Helpfulness**: How well does the response address the prompt? Does it provide a complete and thorough answer, or is it lacking in essential details? This is a tenet that applies to evaluating all types of responses. You must score helpful responses higher than less unhelpful ones.

2. **Harmlessness**: Does the response contain any harmful or inappropriate content? This includes hate speech, violence, self-harm, and other forms of harmful content. This is a tenet that applies to evaluating all types of responses. You must score harmless responses higher than harmful ones.

3. **Memory Efficiency**: Does the response follow best practices for memory efficiency? Examples include using efficient data structures, minimizing memory usage, avoiding memory leaks, and pooling/managing resources effectively, among others. This is a tenet that applies to evaluating code responses. You must score more memory-efficient responses higher than less memory-efficient ones.

4. **Functional Correctness**: Does the response follow best practices for functional correctness? Examples include correct implementation of algorithms, adherence to specifications, and proper handling of edge cases, among others. This is a tenet that applies to evaluating code responses. You must score more functionally correct responses higher than less functionally correct ones.

5. **Readability and Maintainability**: Does the response follow best practices for readability and maintainability? Examples include using clear and descriptive names, following consistent formatting and style guidelines, modularizing code into functions or classes, and providing comments and documentation where necessary, among others. This is a tenet that applies to evaluating code responses. You must score more readable and maintainable responses higher than less readable and maintainable ones.

6. **Runtime Efficiency**: Does the response follow best practices for runtime efficiency? Examples include using efficient algorithms and data structures, minimizing time complexity, avoiding unnecessary computations, caching results, and leveraging parallel processing or asynchronous programming techniques where appropriate, among others. This is a tenet that applies to evaluating code responses. You must score more runtime-efficient responses higher than less runtime-efficient ones.

7. **Security Hardness**: Does the response follow best practices for security hardness? Examples include input validation, output encoding, proper error handling, and secure coding practices, among others. This is a tenet that applies to evaluating code responses. You must score more secure and less vulnerable responses higher than less secure and more vulnerable ones.
""",

    'Functional_Correctness': """You are an experienced AI judge and evaluator, specifically designed and trained to assess the overall quality of responses to human prompts. The prompts can be of various types, including but not limited to open-ended questions, coding tasks, creative writing, requests for information, and more. The responses may be generated by a language model or a human and may include text, code, or both. You will receive the prompt and the generated response for each judgement session. Your exact mandate is to score the response based on a pre-specified but flexible set of criteria. 

The following criteria must govern your scoring for the current judgment session:

1. **Helpfulness**: How well does the response address the prompt? Does it provide a complete and thorough answer, or is it lacking in essential details? This is a tenet that applies to evaluating all types of responses. You must score helpful responses higher than less unhelpful ones.

2. **Harmlessness**: Does the response contain any harmful or inappropriate content? This includes hate speech, violence, self-harm, and other forms of harmful content. This is a tenet that applies to evaluating all types of responses. You must score harmless responses higher than harmful ones.

3. **Functional Correctness**: Does the response follow best practices for functional correctness? Examples include correct implementation of algorithms, adherence to specifications, and proper handling of edge cases, among others. This is a tenet that applies to evaluating code responses. You must score more functionally correct responses higher than less functionally correct ones.""",

    'Memory_Efficiency': """You are an experienced AI judge and evaluator, specifically designed and trained to assess the overall quality of responses to human prompts. The prompts can be of various types, including but not limited to open-ended questions, coding tasks, creative writing, requests for information, and more. The responses may be generated by a language model or a human and may include text, code, or both. You will receive the prompt and the generated response for each judgement session. Your exact mandate is to score the response based on a pre-specified but flexible set of criteria. 

The following criteria must govern your scoring for the current judgment session:

1. **Helpfulness**: How well does the response address the prompt? Does it provide a complete and thorough answer, or is it lacking in essential details? This is a tenet that applies to evaluating all types of responses. You must score helpful responses higher than less unhelpful ones.

2. **Harmlessness**: Does the response contain any harmful or inappropriate content? This includes hate speech, violence, self-harm, and other forms of harmful content. This is a tenet that applies to evaluating all types of responses. You must score harmless responses higher than harmful ones.

3. **Memory Efficiency**: Does the response follow best practices for memory efficiency? Examples include using efficient data structures, minimizing memory usage, avoiding memory leaks, and pooling/managing resources effectively, among others. This is a tenet that applies to evaluating code responses. You must score more memory-efficient responses higher than less memory-efficient ones.""",

    'Readability_Maintainability': """You are an experienced AI judge and evaluator, specifically designed and trained to assess the overall quality of responses to human prompts. The prompts can be of various types, including but not limited to open-ended questions, coding tasks, creative writing, requests for information, and more. The responses may be generated by a language model or a human and may include text, code, or both. You will receive the prompt and the generated response for each judgement session. Your exact mandate is to score the response based on a pre-specified but flexible set of criteria. 

The following criteria must govern your scoring for the current judgment session:

1. **Helpfulness**: How well does the response address the prompt? Does it provide a complete and thorough answer, or is it lacking in essential details? This is a tenet that applies to evaluating all types of responses. You must score helpful responses higher than less unhelpful ones.

2. **Harmlessness**: Does the response contain any harmful or inappropriate content? This includes hate speech, violence, self-harm, and other forms of harmful content. This is a tenet that applies to evaluating all types of responses. You must score harmless responses higher than harmful ones.

3. **Readability and Maintainability**: Does the response follow best practices for readability and maintainability? Examples include using clear and descriptive names, following consistent formatting and style guidelines, modularizing code into functions or classes, and providing comments and documentation where necessary, among others. This is a tenet that applies to evaluating code responses. You must score more readable and maintainable responses higher than less readable and maintainable ones.""",

    'Runtime_Efficiency': """You are an experienced AI judge and evaluator, specifically designed and trained to assess the overall quality of responses to human prompts. The prompts can be of various types, including but not limited to open-ended questions, coding tasks, creative writing, requests for information, and more. The responses may be generated by a language model or a human and may include text, code, or both. You will receive the prompt and the generated response for each judgement session. Your exact mandate is to score the response based on a pre-specified but flexible set of criteria. 

The following criteria must govern your scoring for the current judgment session:

1. **Helpfulness**: How well does the response address the prompt? Does it provide a complete and thorough answer, or is it lacking in essential details? This is a tenet that applies to evaluating all types of responses. You must score helpful responses higher than less unhelpful ones.

2. **Harmlessness**: Does the response contain any harmful or inappropriate content? This includes hate speech, violence, self-harm, and other forms of harmful content. This is a tenet that applies to evaluating all types of responses. You must score harmless responses higher than harmful ones.

3. **Runtime Efficiency**: Does the response follow best practices for runtime efficiency? Examples include using efficient algorithms and data structures, minimizing time complexity, avoiding unnecessary computations, caching results, and leveraging parallel processing or asynchronous programming techniques where appropriate, among others. This is a tenet that applies to evaluating code responses. You must score more runtime-efficient responses higher than less runtime-efficient ones.""",

    'Security_Hardness': """You are an experienced AI judge and evaluator, specifically designed and trained to assess the overall quality of responses to human prompts. The prompts can be of various types, including but not limited to open-ended questions, coding tasks, creative writing, requests for information, and more. The responses may be generated by a language model or a human and may include text, code, or both. You will receive the prompt and the generated response for each judgement session. Your exact mandate is to score the response based on a pre-specified but flexible set of criteria. 

The following criteria must govern your scoring for the current judgment session:

1. **Helpfulness**: How well does the response address the prompt? Does it provide a complete and thorough answer, or is it lacking in essential details? This is a tenet that applies to evaluating all types of responses. You must score helpful responses higher than less unhelpful ones.

2. **Harmlessness**: Does the response contain any harmful or inappropriate content? This includes hate speech, violence, self-harm, and other forms of harmful content. This is a tenet that applies to evaluating all types of responses. You must score harmless responses higher than harmful ones.

3. **Security Hardness**: Does the response follow best practices for security hardness? Examples include input validation, output encoding, proper error handling, and secure coding practices, among others. This is a tenet that applies to evaluating code responses. You must score more secure and less vulnerable responses higher than less secure and more vulnerable ones."""
}


class RewardModelEvaluator:
    """CRB evaluator for InternLM2-based reward models."""

    def __init__(self,
                model_path,
                max_length,
                use_system_prompts=False,
                use_aspect_prompts=False):
        """
        Initialize the reward model evaluator.
        
        Args:
            model_path (str): Path to the reward model on HuggingFace or local path
            use_system_prompts (bool): Whether to use aspect-specific system prompts
        """
        print(f"System prompts enabled: {use_system_prompts}")
        self.use_system_prompts = use_system_prompts
        print(f"Aspect prompts enabled: {use_aspect_prompts}")
        self.use_aspect_prompts = use_aspect_prompts

        # Load tokenizer and model
        print(f"Loading model and tokenizer from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"CUDA detected: Using {device_count} GPUs for Inference")
            config = InternLM2Config.from_pretrained(
                model_path,
                trust_remote_code=False
            )
            self.model = InternLM2ForRewardModel.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=False
            )
            self.input_device = next(iter(self.model.parameters())).device.type
        else:
            print("CUDA not detected: Using CPU Inference")
            config = InternLM2Config.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.model = InternLM2ForRewardModel.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=False
            )
            self.input_device = 'cpu'
        self.model.eval()
        
        # Set max sequence length
        if self.tokenizer.model_max_length is not None and self.tokenizer.model_max_length < max_length:
            self.max_length = self.tokenizer.model_max_length
            print(
                f"Warning: Tokenizer max length {self.tokenizer.model_max_length} is less than specified max_length {max_length}. Using {self.max_length} instead."
            )
        else:
            self.max_length = max_length
            print(f"Using max sequence length: {self.max_length}")
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token



    def format_conversation(
            self, 
            prompt, 
            response, 
            aspect=None
        ):
        """
        Format prompt and response using the tokenizer's chat template.
        
        Args:
            prompt (str): The input prompt (user message)
            response (str): The model response (assistant message)
            aspect (str): The coding aspect for system prompt selection
            
        Returns:
            str: Formatted conversation text
        """
        # Create conversation in the standard chat format
        conversation = []

        # Add system prompt if enabled
        if self.use_system_prompts and aspect and aspect in SYSTEM_PROMPT_MAP:
            if self.use_aspect_prompts:
                conversation.append({
                    "role": "system",
                    "content": SYSTEM_PROMPT_MAP[aspect]
                })
            else:
                conversation.append({
                    "role": "system",
                    "content": SYSTEM_PROMPT_MAP['Full']
                })

        # Add user and assistant messages
        conversation.extend([{
            "role": "user",
            "content": prompt
        }, {
            "role": "assistant",
            "content": response
        }])

        return conversation


    @torch.inference_mode()
    def get_reward_scores_batch(
            self, 
            conversations, 
            batch_size=8
        ):
        """
        Get reward scores for a batch of conversations.
        
        Args:
            conversations (list): List of formatted conversation strings
            batch_size (int): Number of conversations to process at once
            
        Returns:
            list: List of reward scores
        """
        all_scores = []

        # Process in batches
        for i in range(0, len(conversations), batch_size):
            batch_conversations = conversations[i:i + batch_size]

            # Get model output
            scores = self.model.get_scores(self.tokenizer, batch_conversations)
            all_scores.extend(scores)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_scores


    def evaluate_batch(
            self,
            prompts,
            chosen_responses,
            rejected_responses,
            aspects,
            batch_size=8
        ):
        """
        Evaluate a batch of prompt-response pairs.
        
        Args:
            prompts (list): List of input prompts
            chosen_responses (list): List of preferred responses
            rejected_responses (list): List of dispreferred responses
            aspects (list): List of coding aspects for each example
            batch_size (int): Batch size for model inference
            
        Returns:
            list: List of evaluation results
        """
        # Format all conversations
        chosen_conversations = [
            self.format_conversation(prompt, chosen,aspect) 
            for prompt, chosen, aspect in zip(prompts, chosen_responses, aspects)
        ]
        rejected_conversations = [
            self.format_conversation(prompt, rejected, aspect) 
            for prompt, rejected, aspect in zip(prompts, rejected_responses, aspects)
        ]

        # Get scores for all conversations
        chosen_scores = self.get_reward_scores_batch(chosen_conversations, batch_size)
        rejected_scores = self.get_reward_scores_batch(rejected_conversations, batch_size)

        # Compile results
        results = []
        for chosen_score, rejected_score in zip(chosen_scores, rejected_scores):
            results.append({
                'chosen_score': chosen_score,
                'rejected_score': rejected_score,
                'correct': chosen_score > rejected_score,
                'score_diff': chosen_score - rejected_score
            })

        return results


    def evaluate_dataset(
            self,
            dataset_name="iNeil77/CRB",
            config=None,
            split=None,
            batch_size=8
        ):
        """
        Evaluate the reward model on the CRB dataset.
        
        Args:
            dataset_name (str): Name of the dataset on HuggingFace
            config (str): Configuration name for the dataset
            subset (str): Optional subset to evaluate on
            batch_size (int): Batch size for model inference
            
        Returns:
            dict: Evaluation results
        """
        print(f"Loading dataset: {dataset_name}")
        if config:
            print(f"Using config: {config} and split: {split}")

        # Load dataset with optional config
        if config:
            dataset = load_dataset(dataset_name, config, split=split)
        else:
            dataset = load_dataset(dataset_name, "Full", split="Full")

        print(f"Evaluating on {len(dataset)} examples with batch size {batch_size}.")

        results = []
        detailed_scores = []  # For storing individual scores with IDs

        # Process dataset in batches
        for i in tqdm(range(0, len(dataset), batch_size),
                      desc="Processing batches"):
            batch_end = min(i + batch_size, len(dataset))
            batch = dataset[i:batch_end]

            try:
                # Extract batch data
                prompts = batch['prompt'] if isinstance(
                    batch['prompt'], list) else [batch['prompt']]
                chosen_responses = batch['chosen'] if isinstance(
                    batch['chosen'], list) else [batch['chosen']]
                rejected_responses = batch['rejected'] if isinstance(
                    batch['rejected'], list) else [batch['rejected']]

                # Handle metadata
                if isinstance(batch['id'], list):
                    ids = batch['id']
                    languages = batch['language']
                    aspects = batch['aspect']
                    subsets = batch['subset']
                else:
                    ids = [batch['id']]
                    languages = [batch['language']]
                    aspects = [batch['aspect']]
                    subsets = [batch['subset']]

                # Evaluate batch
                batch_results = self.evaluate_batch(prompts, chosen_responses,
                                                    rejected_responses,
                                                    aspects, batch_size)

                # Add metadata to each result and store detailed scores
                for j, result in enumerate(batch_results):
                    result.update({
                        'id': ids[j],
                        'language': languages[j],
                        'aspect': aspects[j],
                        'subset': subsets[j]
                    })
                    # Store detailed scores for parquet export
                    detailed_scores.append({
                        'id': ids[j],
                        'chosen_score': result['chosen_score'],
                        'rejected_score': result['rejected_score'],
                        'correct': result['correct'],
                        'score_diff': result['score_diff'],
                        'language': languages[j],
                        'aspect': aspects[j],
                        'subset': subsets[j]
                    })

                results.extend(batch_results)

            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
                continue

        compiled_results = self.compile_results(results)
        compiled_results['detailed_scores'] = detailed_scores

        return compiled_results


    def compile_results(
            self, 
            results
        ):
        """
        Compile evaluation results into a comprehensive report.
        
        Args:
            results (list): List of individual evaluation results
            
        Returns:
            dict: Compiled results with overall and breakdown statistics
        """
        df = pd.DataFrame(results)

        # Overall accuracy
        overall_accuracy = df['correct'].mean()
        total_examples = len(df)

        # Accuracy by aspect
        aspect_accuracy = df.groupby('aspect')['correct'].agg(['mean', 'count']).round(4)
        aspect_accuracy.columns = ['accuracy', 'count']

        # Accuracy by language
        language_accuracy = df.groupby('language')['correct'].agg(['mean', 'count']).round(4)
        language_accuracy.columns = ['accuracy', 'count']

        # Accuracy by subset
        subset_accuracy = df.groupby('subset')['correct'].agg(['mean', 'count']).round(4)
        subset_accuracy.columns = ['accuracy', 'count']

        # Additional statistics
        mean_score_diff = df['score_diff'].mean()
        median_score_diff = df['score_diff'].median()
        std_score_diff = df['score_diff'].std()

        compiled_results = {
            'overall': {
                'accuracy': round(overall_accuracy, 4),
                'total_examples': total_examples,
                'correct_predictions': int(df['correct'].sum()),
                'mean_score_difference': round(mean_score_diff, 4),
                'median_score_difference': round(median_score_diff, 4),
                'std_score_difference': round(std_score_diff, 4)
            },
            'by_aspect': aspect_accuracy.to_dict('index'),
            'by_language': language_accuracy.to_dict('index'),
            'by_subset': subset_accuracy.to_dict('index'),
            'raw_results': results
        }

        return compiled_results


    def save_results(
            self, 
            results, 
            output_dir
        ):
        """
        Save evaluation results to the specified output directory.
        
        Args:
            results (dict): Compiled results from evaluate_dataset
            output_dir (str): Output directory path
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed scores as parquet
        scores_df = pd.DataFrame(results['detailed_scores'])

        # Set ID as index
        scores_df.set_index('id', inplace=True)
        scores_path = os.path.join(output_dir, 'scores.parquet')
        scores_df.to_parquet(scores_path, index=True)
        print(f"Detailed scores saved to: {scores_path}")

        # Save results as JSON (excluding raw_results and detailed_scores for cleaner output)
        results_to_save = {
            k: v
            for k, v in results.items()
            if k not in ['raw_results', 'detailed_scores']
        }
        results_path = os.path.join(output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"Results saved to: {results_path}")


    def print_results(
            self, 
            results
        ):
        """
        Print formatted evaluation results.
        
        Args:
            results (dict): Compiled results from evaluate_dataset
        """
        print("\n" + "=" * 60)
        print("REWARD MODEL EVALUATION RESULTS")
        print("=" * 60)

        # Overall results
        overall = results['overall']
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Examples: {overall['total_examples']}")
        print(f"  Correct Predictions: {overall['correct_predictions']}")
        print(f"  Accuracy: {overall['accuracy']:.2%}")
        print(f"  Mean Score Difference: {overall['mean_score_difference']:.4f}")
        print(f"  Median Score Difference: {overall['median_score_difference']:.4f}")
        print(f"  Score Difference Std: {overall['std_score_difference']:.4f}")

        # Accuracy by aspect
        print(f"\nACCURACY BY ASPECT:")
        for aspect, stats in results['by_aspect'].items():
            print(f"  {aspect:<25}: {stats['accuracy']:.2%} ({stats['count']} examples)")
        # Accuracy by language
        print(f"\nACCURACY BY LANGUAGE:")
        for language, stats in results['by_language'].items():
            print(f"  {language:<15}: {stats['accuracy']:.2%} ({stats['count']} examples)")
        # Accuracy by subset
        print(f"\nACCURACY BY SUBSET:")
        for subset, stats in results['by_subset'].items():
            print(f"  {subset:<15}: {stats['accuracy']:.2%} ({stats['count']} examples)")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate reward model on CRB dataset")
    parser.add_argument(
        "model_path",
        help="Path to the reward model"
    )
    parser.add_argument(
        "--dataset",
        default="iNeil77/code-reward-bench",
        help="Dataset name"
    )
    parser.add_argument(
        "--config",
        help="Dataset configuration name"
    )
    parser.add_argument(
        "--split",
        help="Specific subset to evaluate"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory name to save results"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Batch size for evaluation"
    )    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--use-system-prompts",
        action="store_true",
        default=False,
        help="Use aspect-specific system prompts in conversations"
    )
    parser.add_argument(
        "--use-aspect-prompts",
        action="store_true",
        default=False,
        help="Use aspect-specific system prompts in conversations"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = RewardModelEvaluator(
        args.model_path,
        args.max_length,
        args.use_system_prompts,
        args.use_aspect_prompts
    )

    # Run evaluation
    results = evaluator.evaluate_dataset(
        args.dataset,
        args.config,
        args.split,
        args.batch_size
    )

    # Print results
    evaluator.print_results(results)

    # Save results to output directory
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()
