"""
Distributed Reward Model Training with Bradley-Terry Objective
==============================================================

Trains a scalar reward model from pairwise human preferences using the
Bradley-Terry framework. Given a prompt x, a preferred response y_w, and a
dispreferred response y_l, the model learns to assign higher scalar rewards
to preferred outputs.

Loss function
-------------
The combined training objective minimises three terms:

  L = -E[ log σ(r_θ(x, y_w) - r_θ(x, y_l))       # (1) Bradley-Terry
         + λ_1 · log p_θ(y_w | x)                  # (2) LM regularisation
         + λ_2 · (|r_θ(x, y_w)| + |r_θ(x, y_l)|)² # (3) Magnitude penalty
       ]

  (1) Preference loss: encourages r(chosen) > r(rejected) via sigmoid margin.
  (2) Language-modelling regularisation: keeps the backbone's generative
      ability intact by training on the chosen completion's next-token loss.
      Computed on the same chosen sequence used for the reward (including
      system prompt if present).
  (3) Magnitude regularisation: prevents unbounded reward growth by penalising
      the squared sum of absolute reward values.

Architecture
------------
- Backbone: any HuggingFace CausalLM loaded via Liger kernel's
  ``AutoLigerKernelForCausalLM`` (applies fused Triton kernels for RMSNorm,
  SwiGLU, RoPE, and cross-entropy throughout the transformer blocks).
- Reward head: a zero-initialised linear projection from the last hidden
  state to a scalar (``nn.Linear(hidden_size, 1, bias=False)``).
- Single forward pass for chosen: the base transformer produces hidden states
  used for both the reward head and (via the LM head) the LM loss.
- Rejected sequences only need the reward, so they use the base transformer
  without the LM head matmul.

Distributed training
--------------------
- Uses HuggingFace Accelerate with FSDP2 (or DeepSpeed) for multi-node,
  multi-GPU training.
- The full loss is computed inside a ``forward()`` method so DDP/FSDP
  gradient-synchronisation hooks fire correctly on the wrapped module.

Checkpointing
-------------
- Saves as ``AutoModelForSequenceClassification`` with ``num_labels=1``
  (scalar regression). The LM head is stripped and the reward head is
  renamed to ``score`` for HuggingFace compatibility.
- Checkpoints can be loaded directly for inference without any custom code.

Dataset format
--------------
Expects the Themis-GeneralPreference schema:
  - ``system``   (str | None) – per-example system prompt
  - ``input``    (str)        – the user query
  - ``chosen``   (str)        – preferred assistant response
  - ``rejected`` (str)        – dispreferred assistant response
  - ``language`` (str)        – language label (optional, for filtering)
  - ``aspect``   (str)        – aspect label (optional, for filtering)

Dependencies
------------
- liger-kernel (required): fused Triton kernels — uses LigerFusedLinearCrossEntropyLoss
  (computes LM loss without materializing the full (B, T, V) logits tensor)
- flash-attn: FlashAttention-2 for memory-efficient attention
- accelerate: distributed training orchestration
- transformers >= 4.40: model loading, chat templates, WSD scheduler
- wandb / tensorboard: experiment tracking

Inference (after training)::

  from transformers import AutoModelForSequenceClassification, AutoTokenizer

  model = AutoModelForSequenceClassification.from_pretrained("./reward_model_output")
  tokenizer = AutoTokenizer.from_pretrained("./reward_model_output")

  inputs = tokenizer("prompt + response text", return_tensors="pt")
  reward = model(**inputs).logits.squeeze()   # scalar reward score

Usage::

  accelerate launch --config_file fsdp2_config.yaml train_reward_model.py \\
      --model_name_or_path Qwen/Qwen3-14B \\
      --dataset_name project-themis/Themis-GeneralPreference \\
      --output_dir ./reward_model_output
"""

import argparse
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)

# ---------------------------------------------------------------------------
# Liger Kernel – required; fused Triton kernels for cross-entropy, RMSNorm, SwiGLU
# ---------------------------------------------------------------------------
try:
    from liger_kernel.transformers import (
        AutoLigerKernelForCausalLM,
        LigerFusedLinearCrossEntropyLoss,
    )
except ImportError:
    raise ImportError(
        "liger-kernel is required but not installed. "
        "Install with: pip install liger-kernel"
    )

logger = get_logger(__name__)


# ============================================================================
# 1. Reward head on top of a causal LM
# ============================================================================
class RewardModelWithLMHead(nn.Module):
    """Reward model built on top of a causal language model backbone.

    Uses a single forward pass through the base transformer for the chosen
    sequence to compute both the scalar reward (from the last hidden state)
    and the LM regularisation loss (by projecting all hidden states through
    the LM head). The rejected sequence only needs a reward, so it uses the
    base transformer without the LM head matmul.

    Attributes:
        backbone: The Liger-patched CausalLM (e.g. LlamaForCausalLM with fused kernels).
        reward_head: Zero-initialised linear layer (hidden_size → 1, no bias).
    """

    def __init__(
        self,
        model_name_or_path: str,
        torch_dtype=torch.bfloat16,
    ):
        super().__init__()

        load_kwargs = dict(
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        )

        self.backbone = AutoLigerKernelForCausalLM.from_pretrained(
            model_name_or_path, **load_kwargs
        )
        self.backbone.config.use_cache = False

        hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.reward_head.weight)

        self._fused_lce = LigerFusedLinearCrossEntropyLoss()

    # ------------------------------------------------------------------
    def _compute_reward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute scalar reward using only the base transformer (no LM head)."""
        outputs = self.backbone.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (B, T, H)
        del outputs
        seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        last_hidden = hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device),
            seq_lengths,
        ]
        return self.reward_head(last_hidden).squeeze(-1)  # (B,)

    # ------------------------------------------------------------------
    def _compute_chosen_reward_and_lm_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass for chosen: returns (reward, lm_loss).

        Runs the base transformer once. The hidden states are used for both
        the reward head (last-token) and the LM head (all tokens, masked to
        completion only).
        """
        outputs = self.backbone.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (B, T, H)
        del outputs

        # --- Reward from last token ---
        seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        last_hidden = hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device),
            seq_lengths,
        ]
        reward = self.reward_head(last_hidden).squeeze(-1)  # (B,)

        # --- LM loss on completion tokens (fused linear CE, never materializes logits) ---
        shift_hidden = hidden_states[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        B, T_shifted = shift_labels.shape
        position_ids = torch.arange(T_shifted, device=input_ids.device).unsqueeze(0)
        is_prompt_or_pad = (
            (position_ids < (prompt_lengths.unsqueeze(1) - 1))
            | (~attention_mask[:, 1:].bool())
        )
        shift_labels = shift_labels.masked_fill(is_prompt_or_pad, -100)

        lm_loss = self._fused_lce(
            self.backbone.lm_head.weight,
            shift_hidden.view(-1, shift_hidden.size(-1)),
            shift_labels.view(-1),
        )

        return reward, lm_loss

    # ------------------------------------------------------------------
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        lambda_lm: float = 0.1,
        lambda_mag: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the full combined training loss.

        Uses a single base-transformer forward pass for chosen (reward + LM loss)
        and a separate reward-only pass for rejected.
        """
        r_chosen, lm_loss = self._compute_chosen_reward_and_lm_loss(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_prompt_lengths"],
        )
        r_rejected = self._compute_reward(
            batch["rejected_input_ids"], batch["rejected_attention_mask"]
        )

        bt_loss = -F.logsigmoid(r_chosen - r_rejected).mean()
        mag_loss = (r_chosen.abs() + r_rejected.abs()).pow(2).mean()
        total_loss = bt_loss + lambda_lm * lm_loss + lambda_mag * mag_loss

        with torch.no_grad():
            r_ch = r_chosen.detach()
            r_rj = r_rejected.detach()
            metrics = {
                "loss/total": total_loss.detach().item(),
                "loss/bt": bt_loss.detach().item(),
                "loss/lm": lm_loss.detach().item(),
                "loss/mag": mag_loss.detach().item(),
                "reward/chosen_mean": r_ch.mean().item(),
                "reward/rejected_mean": r_rj.mean().item(),
                "reward/margin": (r_ch - r_rj).mean().item(),
                "reward/accuracy": (r_ch > r_rj).float().mean().item(),
            }
        return total_loss, metrics


# ============================================================================
# 2. Data collation
# ============================================================================
def build_chat_messages(
    prompt: str,
    response: str,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Construct a message list compatible with HuggingFace chat templates.

    Produces the standard OpenAI-style message format that tokenizer
    ``apply_chat_template`` expects: [{role, content}, ...].

    Args:
        prompt: The user's input query.
        response: The assistant's response (chosen or rejected).
        system_prompt: Optional system-level instruction. If None, the system
            turn is omitted entirely.

    Returns:
        List of message dicts with roles "system" (optional), "user", "assistant".
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": response})
    return messages


# Global counter for truncation logging (across dataset map workers)
_truncation_count = 0


def tokenize_pair(
    example: Dict[str, str],
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
    system_prompt: Optional[str] = None,
    truncate_response: bool = True,
) -> Dict[str, Any]:
    """
    Tokenise a single preference pair using the tokenizer's chat template.

    Compatible with the **Themis-GeneralPreference** schema:
      - ``input``   – the user prompt
      - ``chosen``  – the preferred completion
      - ``rejected``– the dispreferred completion
      - ``system``  – per-example system prompt (may be None)

    The ``system_prompt`` argument acts as a fallback when the example's own
    ``system`` field is empty / missing.

    Response-only truncation
    ~~~~~~~~~~~~~~~~~~~~~~~~
    When ``truncate_response=True`` (default) and a full conversation exceeds
    ``max_length``, only the assistant response is truncated so that the prompt
    (system + user turns with all chat-template special tokens) is always
    preserved in full.  A warning is logged for every truncated example.

    When ``truncate_response=False``, examples that exceed ``max_length`` after
    chat-template tokenization are skipped entirely (returned with a
    ``_skip=True`` sentinel for the caller to filter out).
    """
    global _truncation_count

    prompt_text = example["input"]
    chosen_text = example["chosen"]
    rejected_text = example["rejected"]

    # Per-example system prompt takes priority; CLI flag is the fallback
    effective_system = example.get("system") or system_prompt

    # -- Prompt-only messages (for measuring prompt length) ----------------
    prompt_messages = []
    if effective_system:
        prompt_messages.append({"role": "system", "content": effective_system})
    prompt_messages.append({"role": "user", "content": prompt_text})

    # Tokenise prompt-only with the generation prompt appended so the
    # boundary sits right before the first assistant token.
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
    )
    prompt_len = len(prompt_ids)

    # If the prompt alone already exceeds max_length, we can't fit anything
    if prompt_len >= max_length:
        if truncate_response:
            logger.warning(
                f"Prompt alone ({prompt_len} tokens) exceeds max_length "
                f"({max_length}); skipping example."
            )
        return {"_skip": True}

    # -- Helper: tokenise a response, truncating only the response if needed
    def _tokenise_with_truncation(response_text: str, label: str):
        global _truncation_count

        full_messages = build_chat_messages(prompt_text, response_text, effective_system)
        full_enc = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            return_dict=True,
        )

        full_len = len(full_enc["input_ids"])

        if full_len <= max_length:
            return full_enc

        if not truncate_response:
            return None  # signal to skip

        # -----------------------------------------------------------
        # Truncate only the response: figure out how many response
        # tokens we can keep, re-tokenise with a shorter response.
        # -----------------------------------------------------------
        # Estimate overhead tokens (EOS / closing template tokens)
        # by comparing full-conversation length vs prompt + raw response.
        raw_response_ids = tokenizer.encode(response_text, add_special_tokens=False)
        overhead = full_len - prompt_len - len(raw_response_ids)
        budget = max_length - prompt_len - max(overhead, 0)

        if budget <= 0:
            logger.warning(
                f"No room for response after prompt ({prompt_len} tokens) "
                f"+ overhead ({overhead} tokens) at max_length={max_length}; "
                f"skipping example."
            )
            return None

        # Truncate the raw response token ids, then decode back to text
        truncated_response_ids = raw_response_ids[:budget]
        truncated_response = tokenizer.decode(
            truncated_response_ids, skip_special_tokens=True
        )

        # Re-tokenise with the shortened response through the chat template
        trunc_messages = build_chat_messages(
            prompt_text, truncated_response, effective_system
        )
        trunc_enc = tokenizer.apply_chat_template(
            trunc_messages,
            tokenize=True,
            return_dict=True,
        )

        # Final safety: if still over (due to re-encoding variance), hard-clip
        if len(trunc_enc["input_ids"]) > max_length:
            trunc_enc["input_ids"] = trunc_enc["input_ids"][:max_length]
            trunc_enc["attention_mask"] = trunc_enc["attention_mask"][:max_length]

        _truncation_count += 1
        if _truncation_count <= 50 or _truncation_count % 500 == 0:
            logger.warning(
                f"Truncated {label} response from {full_len} to "
                f"{len(trunc_enc['input_ids'])} tokens (max_length={max_length}). "
                f"Total truncations so far: {_truncation_count}"
            )

        return trunc_enc

    # -- Tokenise chosen & rejected ----------------------------------------
    chosen_enc = _tokenise_with_truncation(chosen_text, "chosen")
    rejected_enc = _tokenise_with_truncation(rejected_text, "rejected")

    if chosen_enc is None or rejected_enc is None:
        return {"_skip": True}

    return {
        "chosen_input_ids": chosen_enc["input_ids"],
        "chosen_attention_mask": chosen_enc["attention_mask"],
        "rejected_input_ids": rejected_enc["input_ids"],
        "rejected_attention_mask": rejected_enc["attention_mask"],
        "chosen_prompt_length": prompt_len,
        "_skip": False,
    }


@dataclass
class PairCollator:
    """DataLoader collate function that pads preference pairs into batched tensors.

    Right-pads chosen and rejected sequences independently to the maximum length
    within the batch (capped at ``max_length``).

    Attributes:
        tokenizer: The tokenizer instance (needed for pad_token_id).
        max_length: Hard ceiling on sequence length after padding.
    """

    tokenizer: AutoTokenizer
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        chosen_ids = [torch.tensor(f["chosen_input_ids"]) for f in features]
        chosen_masks = [torch.tensor(f["chosen_attention_mask"]) for f in features]
        rejected_ids = [torch.tensor(f["rejected_input_ids"]) for f in features]
        rejected_masks = [torch.tensor(f["rejected_attention_mask"]) for f in features]
        prompt_lengths = torch.tensor(
            [f["chosen_prompt_length"] for f in features], dtype=torch.long
        )

        pad_id = self.tokenizer.pad_token_id or 0

        def _pad(tensors, value):
            max_len = min(max(t.size(0) for t in tensors), self.max_length)
            out = torch.full((len(tensors), max_len), value, dtype=tensors[0].dtype)
            for i, t in enumerate(tensors):
                length = min(t.size(0), max_len)
                out[i, :length] = t[:length]
            return out

        return {
            "chosen_input_ids": _pad(chosen_ids, pad_id),
            "chosen_attention_mask": _pad(chosen_masks, 0),
            "rejected_input_ids": _pad(rejected_ids, pad_id),
            "rejected_attention_mask": _pad(rejected_masks, 0),
            "chosen_prompt_lengths": prompt_lengths,
        }


# ============================================================================
# 3. Dataset helpers
# ============================================================================
def prepare_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    max_length: int,
    system_prompt: Optional[str] = None,
    dataset_split: str = "train",
    filter_language: Optional[str] = None,
    filter_aspect: Optional[str] = None,
    truncate_response: bool = True,
    num_proc: int = 8,
):
    """
    Load and tokenise a preference dataset in the
    **Themis-GeneralPreference** format.

    Expected columns
    ----------------
    - ``system``   (str | None) – per-example system prompt
    - ``input``    (str)        – user prompt
    - ``chosen``   (str)        – preferred completion
    - ``rejected`` (str)        – dispreferred completion
    - ``language`` (str)        – language label  (optional, for filtering)
    - ``aspect``   (str)        – aspect label    (optional, for filtering)
    - ``source``   (str)        – provenance tag  (informational)
    - ``idx``      (str)        – unique id       (informational)

    When ``truncate_response=True``, responses that exceed ``max_length``
    (after applying the chat template) are truncated at the response boundary.
    When ``False``, over-length examples are dropped entirely.

    Args:
        num_proc: Number of CPU processes for parallel dataset map/filter
            operations. Defaults to 8. Using all available cores on large
            machines (e.g. 192-core p5) can cause memory pressure and
            contention with the training processes.
    """
    ds = load_dataset(dataset_name, split=dataset_split)

    # ------ Optional filtering by language / aspect ----------------------
    if filter_language is not None:
        ds = ds.filter(
            lambda ex: ex.get("language") == filter_language,
            num_proc=num_proc,
            desc="Filtering by language",
        )
    if filter_aspect is not None:
        ds = ds.filter(
            lambda ex: ex.get("aspect") == filter_aspect,
            num_proc=num_proc,
            desc="Filtering by aspect",
        )

    initial_size = len(ds)

    # ------ Tokenise ------------------------------------------------------
    keep_cols = {
        "chosen_input_ids",
        "chosen_attention_mask",
        "rejected_input_ids",
        "rejected_attention_mask",
        "chosen_prompt_length",
        "_skip",
    }
    ds = ds.map(
        lambda ex: tokenize_pair(
            ex, tokenizer, max_length,
            system_prompt=system_prompt,
            truncate_response=truncate_response,
        ),
        remove_columns=[c for c in ds.column_names if c not in keep_cols],
        num_proc=num_proc,
        desc="Tokenising",
    )

    # Filter out skipped examples (those that couldn't fit max_length)
    pre_filter_size = len(ds)
    ds = ds.filter(
        lambda ex: not ex.get("_skip", False),
        num_proc=num_proc,
        desc="Removing skipped",
    )
    ds = ds.remove_columns(["_skip"])
    skipped = pre_filter_size - len(ds)

    if skipped > 0:
        logger.warning(
            f"Dropped {skipped}/{initial_size} examples that exceeded "
            f"max_length={max_length} "
            f"({'after response truncation' if truncate_response else 'no truncation mode'}). "
            f"Final dataset size: {len(ds)}"
        )

    return ds


# ============================================================================
# 4. Training loop
# ============================================================================
def parse_args():
    """Define and parse all command-line arguments for training configuration.

    Arguments are grouped into:
      - Model: --model_name_or_path
      - Dataset: --dataset_name, --dataset_split, --filter_*, --max_length,
        --truncate_response, --num_proc
      - Loss weights: --lambda_lm, --lambda_mag
      - Optimisation: --learning_rate, --weight_decay, --per_device_train_batch_size,
        --gradient_accumulation_steps, --num_train_epochs
      - LR schedule (WSD): --warmup_ratio, --stable_ratio, --decay_ratio,
        --warmup_type, --decay_type, --min_lr_ratio
      - Checkpointing: --output_dir, --save_steps, --save_epochs
      - Logging: --logging_steps, --report_to, --wandb_project, --wandb_run_name
      - Misc: --bf16, --gradient_checkpointing, --seed, --system_prompt
    """
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--dataset_name", type=str,
                    default="project-themis/Themis-GeneralPreference")
    p.add_argument("--dataset_split", type=str, default="train",
                    help="Dataset split to use (e.g. train, test_rewardbench)")
    p.add_argument("--filter_language", type=str, default=None,
                    help="Only keep examples with this language label "
                         "(e.g. 'Python', 'Java', 'NL')")
    p.add_argument("--filter_aspect", type=str, default=None,
                    help="Only keep examples with this aspect label "
                         "(e.g. 'Helpfulness', 'Harmlessness')")
    p.add_argument("--output_dir", type=str, default="./reward_model_output")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.02,
                    help="Fraction of total steps for warmup phase")
    p.add_argument("--stable_ratio", type=float, default=0.75,
                    help="Fraction of total steps for stable (plateau) phase")
    p.add_argument("--decay_ratio", type=float, default=0.23,
                    help="Fraction of total steps for decay (cooldown) phase. "
                         "warmup_ratio + stable_ratio + decay_ratio should equal 1.0")
    p.add_argument("--warmup_type", type=str, default="linear",
                    choices=["linear", "cosine", "1-sqrt"],
                    help="Shape of the warmup ramp")
    p.add_argument("--decay_type", type=str, default="cosine",
                    choices=["linear", "cosine", "1-sqrt"],
                    help="Shape of the decay curve")
    p.add_argument("--min_lr_ratio", type=float, default=0.0,
                    help="Minimum LR as a ratio of initial LR (0.0 = decay to zero)")
    p.add_argument("--lambda_lm", type=float, default=0.1,
                    help="Weight for LM regularisation on chosen outputs")
    p.add_argument("--lambda_mag", type=float, default=0.01,
                    help="Weight for reward magnitude regularisation")
    p.add_argument("--num_proc", type=int, default=8,
                    help="Number of CPU processes for dataset map/filter operations. "
                         "Defaults to 8. Avoid using all cores on large machines "
                         "(e.g. 192-core p5) to prevent memory pressure.")
    p.add_argument("--system_prompt", type=str, default=None,
                    help="Optional system prompt prepended to every conversation "
                         "via the tokenizer chat template")
    p.add_argument("--truncate_response", action="store_true", default=True,
                    help="Truncate only the response to fit max_length, "
                         "preserving the full prompt (default: enabled)")
    p.add_argument("--no_truncate_response", dest="truncate_response",
                    action="store_false",
                    help="Drop examples that exceed max_length instead of truncating")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--report_to", type=str, default="wandb",
                    choices=["wandb", "tensorboard", "all", "none"],
                    help="Experiment tracker to use")
    p.add_argument("--wandb_project", type=str, default="reward-model-training",
                    help="Weights & Biases project name")
    p.add_argument("--wandb_run_name", type=str, default=None,
                    help="Weights & Biases run name (auto-generated if not set)")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_epochs", action="store_true", default=False,
                    help="Save a checkpoint at the end of each epoch")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    """Entry point: orchestrates the full training pipeline.

    Execution flow:
      1. Parse CLI arguments and initialise the Accelerator (handles distributed
         setup, mixed precision, gradient accumulation, and experiment tracking).
      2. Load the tokenizer and set pad_token if missing.
      3. Instantiate RewardModelWithLMHead with Liger kernels and enable
         gradient checkpointing for memory efficiency.
      4. Load and tokenise the preference dataset (all ranks process independently).
      5. Create AdamW optimiser and Warmup-Stable-Decay (WSD) LR scheduler.
      6. Wrap model/optimiser/dataloader/scheduler with Accelerate for distributed.
      7. Training loop: for each epoch, iterate over batches with gradient
         accumulation, log metrics, and save periodic checkpoints.
      8. Save the final model as AutoModelForSequenceClassification.
    """
    args = parse_args()

    # ---- Accelerator ----
    # Determine which trackers to use
    if args.report_to == "all":
        log_with = ["tensorboard", "wandb"]
    elif args.report_to == "none":
        log_with = None
    else:
        log_with = args.report_to

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16" if args.bf16 else "no",
        log_with=log_with,
        project_dir=args.output_dir,
    )
    set_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # ---- Tokenizer & Model ----
    # Use local_main_process_first so only local rank 0 on each node downloads
    # model weights from HuggingFace Hub. Other processes on the same node wait,
    # then load from the node-local cache.
    with accelerator.local_main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = RewardModelWithLMHead(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        )

    if args.system_prompt and accelerator.is_main_process:
        logger.info(f"Using system prompt: {args.system_prompt!r}")

    if args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # ---- Dataset & DataLoader ----
    dataset = prepare_dataset(
        args.dataset_name,
        tokenizer,
        args.max_length,
        system_prompt=args.system_prompt,
        dataset_split=args.dataset_split,
        filter_language=args.filter_language,
        filter_aspect=args.filter_aspect,
        truncate_response=args.truncate_response,
        num_proc=args.num_proc,
    )
    collator = PairCollator(tokenizer=tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # ---- Optimiser & Scheduler (Warmup-Stable-Decay) ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_update_steps = steps_per_epoch * args.num_train_epochs

    num_warmup_steps = int(args.warmup_ratio * num_update_steps)
    num_stable_steps = int(args.stable_ratio * num_update_steps)
    num_decay_steps = num_update_steps - num_warmup_steps - num_stable_steps

    if accelerator.is_main_process:
        logger.info(
            f"Training: {args.num_train_epochs} epoch(s), "
            f"{steps_per_epoch} steps/epoch, "
            f"{num_update_steps} total update steps"
        )
        logger.info(
            f"WSD schedule: {num_warmup_steps} warmup → "
            f"{num_stable_steps} stable → {num_decay_steps} decay"
        )

    lr_scheduler = get_scheduler(
        "warmup_stable_decay",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_update_steps,
        scheduler_specific_kwargs={
            "num_stable_steps": num_stable_steps,
            "num_decay_steps": num_decay_steps,
            "warmup_type": args.warmup_type,
            "decay_type": args.decay_type,
            "min_lr_ratio": args.min_lr_ratio,
        },
    )

    # ---- Prepare for distributed ----
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # ---- Tracking ----
    if accelerator.is_main_process:
        tracker_init_kwargs = {}
        if args.report_to in ("wandb", "all"):
            tracker_init_kwargs["wandb"] = {
                "name": args.wandb_run_name,
                "config": vars(args),
            }
        accelerator.init_trackers(
            args.wandb_project,
            init_kwargs=tracker_init_kwargs,
        )

    # ---- Train ----
    global_step = 0
    for epoch in range(args.num_train_epochs):
        model.train()

        if accelerator.is_main_process:
            logger.info(f"===== Epoch {epoch + 1}/{args.num_train_epochs} =====")

        # Accelerate wraps the dataloader with a DistributedSampler that
        # supports set_epoch for per-epoch reshuffling.
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch)
        elif hasattr(dataloader.batch_sampler, "sampler") and hasattr(
            dataloader.batch_sampler.sampler, "set_epoch"
        ):
            dataloader.batch_sampler.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                loss, metrics = model(
                    batch,
                    lambda_lm=args.lambda_lm,
                    lambda_mag=args.lambda_mag,
                )
                accelerator.backward(loss)
                del loss
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                epoch_steps += 1
                epoch_loss += metrics["loss/total"]

                if global_step % args.logging_steps == 0:
                    metrics["train/learning_rate"] = lr_scheduler.get_last_lr()[0]
                    metrics["train/epoch"] = epoch + (epoch_steps / steps_per_epoch)
                    accelerator.log(metrics, step=global_step)
                    if accelerator.is_main_process:
                        logger.info(
                            f"epoch {epoch + 1} | step {global_step} | "
                            f"loss {metrics['loss/total']:.4f} | "
                            f"bt {metrics['loss/bt']:.4f} | "
                            f"lm {metrics['loss/lm']:.4f} | "
                            f"mag {metrics['loss/mag']:.4f} | "
                            f"acc {metrics['reward/accuracy']:.3f} | "
                            f"margin {metrics['reward/margin']:.3f}"
                        )

                if global_step % args.save_steps == 0:
                    _save_checkpoint(
                        accelerator, model, tokenizer,
                        os.path.join(args.output_dir, f"checkpoint-{global_step}"),
                    )

        # End-of-epoch logging and optional checkpoint
        if accelerator.is_main_process:
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            logger.info(
                f"Epoch {epoch + 1}/{args.num_train_epochs} complete. "
                f"Avg loss: {avg_epoch_loss:.4f}, "
                f"Steps this epoch: {epoch_steps}, "
                f"Global step: {global_step}"
            )

        if args.save_epochs:
            _save_checkpoint(
                accelerator, model, tokenizer,
                os.path.join(args.output_dir, f"epoch-{epoch + 1}"),
            )

    # ---- Final save ----
    _save_checkpoint(accelerator, model, tokenizer, args.output_dir)
    if accelerator.is_main_process:
        logger.info(f"Model saved to {args.output_dir}")

    accelerator.end_training()


def _save_checkpoint(
    accelerator: Accelerator,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    save_dir: str,
) -> None:
    """
    Save model weights as an ``AutoModelForSequenceClassification``-compatible
    checkpoint with ``num_labels=1`` (scalar reward).

    The saved checkpoint can be loaded at inference time with::

        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        tokenizer = AutoTokenizer.from_pretrained(save_dir)

    Key steps:

    1. Gather the full state dict (handles FSDP2 all-gather + CPU offload).
    2. Strip the ``backbone.`` prefix from all keys.
    3. Remove the causal LM head (``lm_head.*``) — it is not needed for
       reward inference and would bloat the checkpoint.
    4. Rename ``reward_head.weight`` → ``score.weight`` to match the
       ``*ForSequenceClassification`` architecture.
    5. Save a config with ``num_labels=1`` and the correct
       ``*ForSequenceClassification`` auto-mapping so that
       ``AutoModelForSequenceClassification.from_pretrained`` works.
    """
    accelerator.wait_for_everyone()

    # get_state_dict gathers the full state dict to rank 0 under FSDP
    full_state_dict = accelerator.get_state_dict(model)

    unwrapped = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Remap keys: strip "backbone." prefix, drop LM head, rename reward
        # head to "score" (the name used by *ForSequenceClassification).
        # ------------------------------------------------------------------
        seq_cls_state = {}
        for key, value in full_state_dict.items():
            # Strip the wrapping module prefix
            if key.startswith("backbone."):
                inner_key = key[len("backbone."):]
            else:
                inner_key = key

            # Drop the causal LM head entirely
            if inner_key.startswith("lm_head."):
                continue

            # Rename reward_head → score (the classification head name)
            if inner_key.startswith("reward_head."):
                inner_key = inner_key.replace("reward_head.", "score.", 1)

            seq_cls_state[inner_key] = value

        # ------------------------------------------------------------------
        # Build a config for *ForSequenceClassification with num_labels=1
        # ------------------------------------------------------------------
        config = AutoConfig.from_pretrained(unwrapped.backbone.config._name_or_path)
        config.num_labels = 1
        config.problem_type = "regression"

        # Ensure pad_token_id is set so the model can find the last token
        if tokenizer.pad_token_id is not None:
            config.pad_token_id = tokenizer.pad_token_id

        # Instantiate a *ForSequenceClassification shell on CPU (meta-init
        # would be cleaner but not all architectures support it, so we just
        # use the config to resolve the correct class and save directly).
        seq_cls_model = AutoModelForSequenceClassification.from_config(config)

        # Load our remapped weights into the shell
        missing, unexpected = seq_cls_model.load_state_dict(seq_cls_state, strict=False)
        if missing:
            logger.warning(f"Keys missing when saving as SeqCls: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when saving as SeqCls: {unexpected}")

        # Save everything: safetensors weights, config.json, tokenizer files
        seq_cls_model.save_pretrained(save_dir, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)

        logger.info(
            f"Checkpoint saved as AutoModelForSequenceClassification "
            f"(num_labels=1) → {save_dir}"
        )


if __name__ == "__main__":
    main()
