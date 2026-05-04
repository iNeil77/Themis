---
library_name: transformers
tags:
- code
- reward-model
- multilingual
license: apache-2.0
datasets:
- project-themis/Themis-CodePreference
base_model:
- project-themis/Themis-RM-32B-PMP
pipeline_tag: text-classification
language:
- en
---

<div align="center">

# Themis-RM-32B

[![arXiv](https://img.shields.io/badge/arXiv-2605.00754-b31b1b.svg)](https://arxiv.org/abs/2605.00754)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-Themis--RM-yellow)](https://huggingface.co/collections/project-themis/themis-reward-model-collection)
[![Datasets & Benchmarks](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets%20%26%20Benchmarks-Themis-blue)](https://huggingface.co/collections/project-themis/themis-preference-datasets-and-benchmarks)
[![GitHub](https://img.shields.io/badge/GitHub-Themis-181717?logo=github)](https://github.com/iNeil77/Themis)
[![Docker](https://img.shields.io/badge/Docker-ineil77%2Fthemis-2496ED?logo=docker)](https://hub.docker.com/repository/docker/ineil77/themis/general)

</div>

## Overview

**Themis-RM-32B** is a 32B-parameter multilingual code reward model for flexible multi-criteria scoring. It is the flagship model of the [Themis-RM](https://huggingface.co/collections/project-themis/themis-reward-model-collection) suite, trained using the Bradley-Terry preference framework on [Themis-CodePreference](https://huggingface.co/collections/project-themis/themis-preference-datasets-and-benchmarks), the largest open-source collection of code preferences to date (more than 350k preference pairs).

Themis-RM models evaluate code across five quality dimensions — Functional Correctness, Runtime Efficiency, Memory Efficiency, Security Hardness, and Readability & Maintainability — and support eight programming languages. Our experiments demonstrate positive scaling trends, strong cross-lingual transfer when training on diverse preferences, and the importance of multi-criteria training for reliable code reward modelling.

## Model Family

The Themis-RM suite ranges from 600M to 32B parameters, all built on the Qwen3 backbone.

<div align="center">

| Model | Model Architecture | HuggingFace Model Page |
|:---|:---:|:---:|
| Themis-RM-0.6B | <a href="https://huggingface.co/Qwen/Qwen3-0.6B" style="white-space: nowrap"><span><img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" width="16" style="vertical-align: middle"> Qwen/Qwen3-0.6B</span></a> | <a href="https://huggingface.co/project-themis/Themis-RM-0.6B" style="white-space: nowrap"><span><img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" width="16" style="vertical-align: middle"> Themis-RM-0.6B</span></a> |
| Themis-RM-1.7B | <a href="https://huggingface.co/Qwen/Qwen3-1.7B" style="white-space: nowrap"><span><img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" width="16" style="vertical-align: middle"> Qwen/Qwen3-1.7B</span></a> | <a href="https://huggingface.co/project-themis/Themis-RM-1.7B" style="white-space: nowrap"><span><img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" width="16" style="vertical-align: middle"> Themis-RM-1.7B</span></a> |
| Themis-RM-4B | <a href="https://huggingface.co/Qwen/Qwen3-4B" style="white-space: nowrap"><span><img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" width="16" style="vertical-align: middle"> Qwen/Qwen3-4B</span></a> | <a href="https://huggingface.co/project-themis/Themis-RM-4B" style="white-space: nowrap"><span><img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" width="16" style="vertical-align: middle"> Themis-RM-4B</span></a> |
| Themis-RM-8B | <a href="https://huggingface.co/Qwen/Qwen3-8B" style="white-space: nowrap"><span><img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" width="16" style="vertical-align: middle"> Qwen/Qwen3-8B</span></a> | <a href="https://huggingface.co/project-themis/Themis-RM-8B" style="white-space: nowrap"><span><img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" width="16" style="vertical-align: middle"> Themis-RM-8B</span></a> |
| Themis-RM-14B | <a href="https://huggingface.co/Qwen/Qwen3-14B" style="white-space: nowrap"><span><img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" width="16" style="vertical-align: middle"> Qwen/Qwen3-14B</span></a> | <a href="https://huggingface.co/project-themis/Themis-RM-14B" style="white-space: nowrap"><span><img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" width="16" style="vertical-align: middle"> Themis-RM-14B</span></a> |
| **Themis-RM-32B (this model)** | <a href="https://huggingface.co/Qwen/Qwen3-32B" style="white-space: nowrap"><span><img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" width="16" style="vertical-align: middle"> Qwen/Qwen3-32B</span></a> | — |

</div>

## Results

Themis-RM models achieve best-in-class accuracy on [Themis-CodeRewardBench](https://huggingface.co/datasets/project-themis/Themis-CodeRewardBench), a code-specific reward model benchmark, while also matching or exceeding much larger models on established general-domain benchmarks ([RewardBench V1](https://huggingface.co/datasets/allenai/reward-bench), [RewardBench V2](https://huggingface.co/datasets/allenai/reward-bench-v2), [JudgeBench](https://huggingface.co/datasets/ScalerLab/JudgeBench)). Models are grouped by parameter class; **bold** marks the best in each group.

<div align="center">

| Model | [Themis-CodeRewardBench](https://huggingface.co/datasets/project-themis/Themis-CodeRewardBench) | [RewardBench V1](https://huggingface.co/datasets/allenai/reward-bench) | [RewardBench V2](https://huggingface.co/datasets/allenai/reward-bench-v2) | [JudgeBench](https://huggingface.co/datasets/ScalerLab/JudgeBench) |
|---|---|---|---|---|
| | | | | |
| **32B - 72B Class** | | | | |
| [WorldPM-72B](https://huggingface.co/Qwen/WorldPM-72B-RLHFLow) | 76.96 | 90.88 | 67.92 | 55.21 |
| [Athene-RM-70B](https://huggingface.co/Nexusflow/Athene-RM-70B) | 78.39 | 91.22 | 68.76 | 63.45 |
| [Nemotron-70B-Reward](https://huggingface.co/nvidia/Llama-3.3-Nemotron-70B-Reward) | 81.19 | 93.88 | 70.49 | **73.47** |
| **Themis-RM-32B (this model)** | **91.82** | **94.89** | **72.34** | 71.65 |
| [AceCodeRM-32B](https://huggingface.co/TIGER-Lab/AceCodeRM-32B) | 62.95 | 23.58 | 67.98 | 66.77 |
| | | | | |
| **7B - 14B Class** | | | | |
| [**Themis-RM-14B**](https://huggingface.co/project-themis/Themis-RM-14B) | **91.19** | 94.11 | 71.44 | **70.85** |
| [**Themis-RM-8B**](https://huggingface.co/project-themis/Themis-RM-8B) | 89.78 | 93.69 | 65.87 | 69.97 |
| [Athene-RM-8B](https://huggingface.co/Nexusflow/Athene-RM-8B) | 76.58 | 87.48 | 62.96 | 61.12 |
| [CodeScaler-8B](https://huggingface.co/LARK-Lab/CodeScaler-8B) | 79.12 | 94.66 | 76.51 | 70.05 |
| [Skywork-Reward-V2-8B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B) | 79.97 | **94.76** | **76.93** | 67.90 |
| [AceCodeRM-7B](https://huggingface.co/TIGER-Lab/AceCodeRM-7B) | 71.11 | 22.74 | 63.16 | 61.09 |
| | | | | |
| **0.6B - 4B Class** | | | | |
| [**Themis-RM-4B**](https://huggingface.co/project-themis/Themis-RM-4B) | **88.39** | 92.46 | 63.81 | 68.02 |
| [CodeScaler-4B](https://huggingface.co/LARK-Lab/CodeScaler-4B) | 77.97 | **94.32** | **75.13** | **68.44** |
| [Skywork-Reward-V2-4B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-4B) | 79.27 | 94.06 | 74.26 | 65.43 |
| [**Themis-RM-1.7B**](https://huggingface.co/project-themis/Themis-RM-1.7B) | 83.04 | 89.17 | 56.22 | 63.29 |
| [CodeScaler-1.7B](https://huggingface.co/LARK-Lab/CodeScaler-1.7B) | 73.75 | 91.13 | 68.44 | 66.17 |
| [Skywork-Reward-V2-1.7B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-1.7B) | 75.60 | 91.64 | 67.71 | 66.48 |
| [**Themis-RM-0.6B**](https://huggingface.co/project-themis/Themis-RM-0.6B) | 79.26 | 83.41 | 49.61 | 63.84 |
| [Skywork-Reward-V2-0.6B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B) | 72.77 | 86.32 | 60.83 | 63.65 |

</div>

## Usage

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "project-themis/Themis-RM-32B"
device = "cuda:0"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
    num_labels=1,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Write a Python function that checks if a string is a palindrome."

response_chosen = """def is_palindrome(s: str) -> bool:
    s = s.lower().strip()
    return s == s[::-1]"""

response_rejected = """def is_palindrome(s: str) -> bool:
    for i in range(len(s)):
        if s[i] != s[len(s) - i]:
            return False
    return True"""

conv_chosen = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": response_chosen},
]
conv_rejected = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": response_rejected},
]

chosen_text = tokenizer.apply_chat_template(conv_chosen, tokenize=False)
rejected_text = tokenizer.apply_chat_template(conv_rejected, tokenize=False)

inputs_chosen = tokenizer(chosen_text, return_tensors="pt", truncation=True, max_length=4096).to(device)
inputs_rejected = tokenizer(rejected_text, return_tensors="pt", truncation=True, max_length=4096).to(device)

with torch.no_grad():
    score_chosen = model(**inputs_chosen).logits[0][0].item()
    score_rejected = model(**inputs_rejected).logits[0][0].item()

print(f"Chosen response score:   {score_chosen}")
print(f"Rejected response score: {score_rejected}")
```

### Multi-Criteria Scoring with System Prompts

Themis-RM models are trained with stochastic criteria-conditioned system prompts, allowing you to steer scoring toward a specific quality dimension at inference time. Prepend a system message that specifies the evaluation criteria before the user/assistant turns. The model supports the following criteria:

<div align="center">

| Criterion | Key |
|:---|:---|
| Functional Correctness | `Functional_Correctness` |
| Runtime Efficiency | `Runtime_Efficiency` |
| Memory Efficiency | `Memory_Efficiency` |
| Security Hardness | `Security_Hardness` |
| Readability & Maintainability | `Readability_Maintainability` |
| All criteria (multi-criteria) | `Full` |

</div>

Each criterion-specific system prompt includes a base preamble (Helpfulness + Harmlessness) followed by the targeted criterion. The `Full` prompt includes all five code criteria together. When no system prompt is provided, the model scores based on general quality — the model was trained to handle all three modes (no system prompt, single-criterion, and full multi-criteria). For the full set of system prompts used during training and evaluation, see the [Evaluation](https://github.com/iNeil77/Themis/tree/main/Evaluation) folder in the GitHub repository.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "project-themis/Themis-RM-32B"
device = "cuda:0"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
    num_labels=1,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Criterion-specific system prompt (example: Runtime Efficiency)
# See https://github.com/iNeil77/Themis/tree/main/Evaluation for all prompts
system_prompt = (
    "You are an experienced AI judge and evaluator, specifically "
    "designed and trained to assess the overall quality of responses "
    "to human prompts. The prompts can be of various types, including "
    "but not limited to open-ended questions, coding tasks, creative "
    "writing, requests for information, and more. The responses may be "
    "generated by a language model or a human and may include text, "
    "code, or both. You will receive the prompt and the generated "
    "response for each judgement session. Your exact mandate is to "
    "score the response based on a pre-specified but flexible set of "
    "criteria.\n\n"
    "The following criteria must govern your scoring for the current "
    "judgment session:\n\n"
    "1. **Helpfulness**: How well does the response address the "
    "prompt? Does it provide a complete and thorough answer, or is it "
    "lacking in essential details? This is a tenet that applies to "
    "evaluating all types of responses. You must score helpful "
    "responses higher than less unhelpful ones.\n\n"
    "2. **Harmlessness**: Does the response contain any harmful or "
    "inappropriate content? This includes hate speech, violence, "
    "self-harm, and other forms of harmful content. This is a tenet "
    "that applies to evaluating all types of responses. You must "
    "score harmless responses higher than harmful ones.\n\n"
    "3. **Runtime Efficiency**: Does the response follow best "
    "practices for runtime efficiency? Examples include using "
    "efficient algorithms and data structures, minimizing time "
    "complexity, avoiding unnecessary computations, caching results, "
    "and leveraging parallel processing or asynchronous programming "
    "techniques where appropriate, among others. This is a tenet "
    "that applies to evaluating code responses. You must score more "
    "runtime-efficient responses higher than less runtime-efficient "
    "ones."
)

prompt = "Write a Python function that returns the n-th Fibonacci number."

response = """def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b"""

conversation = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": response},
]

text = tokenizer.apply_chat_template(conversation, tokenize=False)
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)

with torch.no_grad():
    score = model(**inputs).logits[0][0].item()

print(f"Runtime Efficiency score: {score}")
```

## License

This model is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). The base model, [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B), is also licensed under Apache 2.0.

## Citation

```bibtex
@article{themis2025,
  title={Themis: Training Robust Multilingual Code Reward Models for Flexible Multi-Criteria Scoring},
  author={},
  journal={arXiv preprint arXiv:2605.00754},
  year={2025}
}
```
