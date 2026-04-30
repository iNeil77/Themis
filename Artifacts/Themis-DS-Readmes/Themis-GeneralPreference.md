---
tags:
- reward-model
- preference
- multilingual
license: apache-2.0
task_categories:
- text-classification
language:
- en
size_categories:
- 100K<n<1M
---

<div align="center">

# Themis-GeneralPreference

[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-Themis--RM-yellow)](https://huggingface.co/collections/project-themis/themis-reward-model-collection)
[![Datasets & Benchmarks](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets%20%26%20Benchmarks-Themis-blue)](https://huggingface.co/collections/project-themis/themis-preference-datasets-and-benchmarks)
[![GitHub](https://img.shields.io/badge/GitHub-Themis-181717?logo=github)](https://github.com/iNeil77/Themis)

</div>

## Overview

**Themis-GeneralPreference** is a 110k+ sample mix of natural language and code preferences curated from popular existing preference and retrieval datasets. It serves as the training dataset for the preference model pre-training (PT) stage of the [Themis-RM](https://huggingface.co/collections/project-themis/themis-reward-model-collection) suite, designed to instill common human-inspired notions of preference evaluation such as relevance, helpfulness, and harmlessness before the code-specialized preference modeling stage.

This approach mirrors prior work demonstrating the benefits of training LMs for approximating open-domain human preferences. The PT stage is followed by the PM stage on [Themis-CodePreference](https://huggingface.co/datasets/project-themis/Themis-CodePreference), which specializes the model for multi-criteria code scoring.

## Dataset Composition

<div align="center">

| Sub-Dataset | Samples | Criteria | Languages | Source |
|:---|:---:|:---|:---|:---|
| **CodeR-Pile** | 41,924 | Helpfulness | C, C#, C++, Go, Java, JS, Python, Ruby (code) | [nebula2025/CodeR-Pile](https://huggingface.co/datasets/nebula2025/CodeR-Pile) |
| **Skywork-Preference** | 30,146 | Helpfulness, Harmlessness | Natural Language | [Skywork/Skywork-Reward-Preference-80K-v0.2](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2) |
| **Tulu-IF** | 13,705 | Helpfulness | Natural Language | [allenai/tulu-3-pref-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-pref-personas-instruction-following) |
| **H4-Stackexchange** | 12,139 | Helpfulness | Natural Language | [HuggingFaceH4/stack-exchange-preferences](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences) |
| **Arena-HumanPreference** | 7,068 | Helpfulness | Natural Language | [lmarena-ai/arena-human-preference-140k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) |
| **Prometheus-Preference** | 2,864 | Helpfulness, Harmlessness | Natural Language | [prometheus-eval/Preference-Collection](https://huggingface.co/datasets/prometheus-eval/Preference-Collection) |
| **HelpSteer3** | 1,452 | Helpfulness | Natural Language | [nvidia/HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) |
| **Argilla-DPO** | 1,042 | Helpfulness | Natural Language | [argilla/distilabel-math-preference-dpo](https://huggingface.co/datasets/argilla/distilabel-math-preference-dpo) |
| **Truthy-DPO** | 377 | Helpfulness | Natural Language | [jondurbin/truthy-dpo-v0.1](https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1) |

</div>

**CodeR-Pile** — Preference pairs from the CodeR-Pile code retrieval dataset (augmentation, exemplar, integration, refinement, simplification, pseudo-code, tutorial, and web query subsets). Chosen from positive document, rejected from Zipfian-sampled hard negatives.

**Skywork-Preference** — Preference pairs from Skywork Preference (WildGuard, OffsetBias, Magpie, and HelpSteer2 subsets).

**Tulu-IF** — Response pairs that follow and violate pre-specified instructions from the Tulu v3 instruction-tuning dataset.

**H4-Stackexchange** — Filtered StackExchange technical forums — highly-voted accepted answers as chosen, well-formed but low-scoring answers as rejected.

**Arena-HumanPreference** — Human voting-validated preferences from filtered LMArena model head-to-head contest data.

**Prometheus-Preference** — Response pairs from instruction clusters for helpfulness and harmlessness from Prometheus Preference data.

**HelpSteer3** — High-margin preference pairs from HelpSteer3 Preference (general and stem subsets).

**Argilla-DPO** — High-margin math answer preference pairs. Synthetically filtered for pairs where both chosen and rejected converge on the same answer, selecting for stylistic preferences beyond correctness.

**Truthy-DPO** — Synthetically labeled truthfulness preferences mined from existing instruction-tuning data.

## Filtering Procedure

The dataset undergoes thorough cleaning and decontamination:

<div align="center">

| Step | Filter | Details |
|:---:|:---|:---|
| 1 | **Max token length** | 2,560 tokens (as measured by the Themis-RM tokenizer) |
| 2 | **Code syntax filtering** | Code responses with AST depth < 3 filtered out |
| 3 | **Language filtering** | Non-English prompts discarded via GlotLID classifier |
| 4 | **Perplexity filtering** | Prompts with perplexity > 1,200 discarded (KenLM model on OSCAR EN corpus) |
| 5 | **Near-deduplication** | MinHash filter (shingle size 20, similarity threshold 0.75) |
| 6 | **Benchmark decontamination** | 13-gram overlap removal against Themis-CodeRewardBench, RewardBench, and RM-Bench |

</div>

## Training Usage

This dataset is used for the **PT (preference model pre-training)** stage of Themis-RM training, the first of two training stages. During training, stochastic system prompts are assigned:

<div align="center">

| Probability | System Prompt Strategy |
|:---:|:---|
| 15% | No system prompt |
| 60% | Aspect-specific prompt (Helpfulness + Harmlessness + target criterion) |
| 25% | Full multi-criteria prompt (all 5 code criteria) |

</div>

Training uses the **Bradley-Terry objective** with auxiliary conditional language modeling loss and reward magnitude regularization. Two epochs, sequence length 2,560, global batch size 1,024, AdamW optimizer with cosine scheduler and 5% warmup.

For full training details and system prompt construction, see the [Dataset](https://github.com/iNeil77/Themis/tree/main/Dataset) folder in the GitHub repository.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("project-themis/Themis-GeneralPreference")

sample = dataset["train"][0]
print(f"Prompt: {sample['prompt'][:100]}...")
print(f"Chosen length: {len(sample['chosen'])}")
print(f"Rejected length: {len(sample['rejected'])}")
```

## License

This dataset is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

## Citation

```bibtex
@article{themis2025,
  title={Themis: Training Robust Multilingual Code Reward Models for Flexible Multi-Criteria Scoring},
  author={Paul, Indraneil and Gurevych, Iryna and Glava\v{s}, Goran},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```
