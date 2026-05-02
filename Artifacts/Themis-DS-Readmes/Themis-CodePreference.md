---
dataset_info:
  features:
  - name: system
    dtype: large_string
  - name: input
    dtype: large_string
  - name: chosen
    dtype: large_string
  - name: rejected
    dtype: large_string
  - name: language
    dtype:
      class_label:
        names:
          '0': JavaScript
          '1': Python
          '2': Ruby
          '3': C++
          '4': C
          '5': Go
          '6': Java
          '7': C#
          '8': NL
  - name: aspect
    dtype:
      class_label:
        names:
          '0': Readability and Maintainability
          '1': Runtime Efficiency
          '2': Security Hardness
          '3': Functional Correctness
          '4': Memory Efficiency
          '5': Helpfulness
          '6': Harmlessness
  - name: source
    dtype: string
  - name: idx
    dtype: string
  splits:
  - name: train
    num_bytes: 3018930512
    num_examples: 354010
  download_size: 1027355104
  dataset_size: 3018930512
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
tags:
- code
- reward-model
- preference
- multilingual
license: apache-2.0
task_categories:
- text-classification
language:
- code
size_categories:
- 100K<n<1M
---


<div align="center">

# Themis-CodePreference

[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-Themis--RM-yellow)](https://huggingface.co/collections/project-themis/themis-reward-model-collection)
[![Datasets & Benchmarks](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets%20%26%20Benchmarks-Themis-blue)](https://huggingface.co/collections/project-themis/themis-preference-datasets-and-benchmarks)
[![GitHub](https://img.shields.io/badge/GitHub-Themis-181717?logo=github)](https://github.com/iNeil77/Themis)
[![Docker](https://img.shields.io/badge/Docker-ineil77%2Fthemis-2496ED?logo=docker)](https://hub.docker.com/repository/docker/ineil77/themis/general)

</div>

## Overview

**Themis-CodePreference** is the largest open-source collection of code preferences to date, containing more than 350k preference pairs. It is the primary training dataset for the preference modeling (PM) stage of the [Themis-RM](https://huggingface.co/collections/project-themis/themis-reward-model-collection) suite of multilingual code reward models. The dataset covers five dimensions of code quality — **Functional Correctness**, **Runtime Efficiency**, **Memory Efficiency**, **Security Hardness**, and **Readability & Maintainability** — across eight programming languages: C, C#, C++, Go, Java, JavaScript, Python, and Ruby.

The dataset centers on sourcing diverse preference scenarios from GitHub commits (mined from [Themis-Git-Commits](https://huggingface.co/datasets/project-themis/git-commits) via [Themis-Git-Commits-Merged](https://huggingface.co/datasets/project-themis/git-commits-merged)) and from synthetically bugged instruction-tuning data, augmented with training sets from pre-existing code-preference and retrieval datasets.

## Dataset Composition

The dataset comprises 11 constituent sub-datasets grouped into four categories. Criteria abbreviations: **FC** = Functional Correctness, **RT** = Runtime Efficiency, **Mem** = Memory Efficiency, **Sec** = Security Hardness, **Read** = Readability & Maintainability.

### Commit-Based Preferences

<div align="center">

| Sub-Dataset | Samples | Criteria | Languages | Source |
|:---|:---:|:---|:---|:---|
| **Commit-Preference** | 126,586 | FC, RT, Mem, Read, Sec | C, C#, C++, Go, Java, JS, Python, Ruby | Code preferences from non-reverted single-file GitHub commits mined from **permissively licensed** repositories (via a modified [OctoPack](https://arxiv.org/abs/2308.07124)/[CommitPack](https://huggingface.co/datasets/bigcode/commitpack) pipeline). The BigQuery snapshot contains commits up to early 2022 (pre-LLM era), guaranteeing human-authored code. Commits verified to be part of successfully-merged pull requests and classified by criteria using [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) commit classifiers trained on [curated term lists](https://github.com/iNeil77/Themis/tree/main/Dataset/Commit_Mining_Terms). Preference strength validated via multi-LLM consensus. All commits pushed before **March 2019** from disjoint repositories vs. benchmark data. |

</div>

### Retrieval & Instruction-Based Preferences

<div align="center">

| Sub-Dataset | Samples | Criteria | Languages | Source |
|:---|:---:|:---|:---|:---|
| **CodeR-Pile** | 77,153 | FC | C, C#, C++, Go, Java, JS, Python, Ruby | Preference pairs from [CodeR-Pile](https://huggingface.co/datasets/nebula2025/CodeR-Pile). Chosen from positive document code, rejected from Zipfian-sampled hard negatives. |
| **Bugged-Instruct** | 54,969 | FC | C, C#, C++, Go, Java, JS, Python, Ruby | Instruction-tuning datasets repurposed by creating rejected responses via model-based bug introduction. Sources include OSS-Instruct, Inverse-Instruct, McEval-Instruct, and Package-Instruct. |

</div>

### Security Preferences

<div align="center">

| Sub-Dataset | Samples | Criteria | Languages | Source |
|:---|:---:|:---|:---|:---|
| **ProSec** | 37,690 | Sec | C, C++, Java, JS, Python | Synthetically generated vulnerability-fix pairs from [ProSec](https://huggingface.co/datasets/prosecalign/prosec-mixed-clm7b-inst). |
| **Cybernative-DPO** | 1,354 | Sec | C#, C++, Go, Java, JS, Python, Ruby | Security-fix preferences via synthetic fixes to vulnerable code from [CyberNative](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO). |

</div>

### Efficiency & Correctness Preferences

<div align="center">

| Sub-Dataset | Samples | Criteria | Languages | Source |
|:---|:---:|:---|:---|:---|
| **Venus** | 21,784 | RT, Mem | Python | Runtime and memory usage preferences from [Venus](https://huggingface.co/datasets/Elfsong/Venus), selecting samples where chosen is at least 5x more efficient. |
| **CodeNet** | 16,819 | RT, Mem | C, C#, C++, Go, Java, JS, Python, Ruby | Runtime and memory preferences per submitter from [CodeNet](https://huggingface.co/datasets/iNeil77/CodeNet), with 5x efficiency threshold. |
| **RunBugRun** | 6,931 | FC | Python | Code contest program-repair pairs from the [RunBugRun](https://huggingface.co/datasets/ASSERT-KTH/RunBugRun-Final) train set. |
| **ECCO** | 6,188 | RT | Python | Code contest runtime preferences from the [ECCO](https://huggingface.co/datasets/CodeEff/ECCO) train set. |
| **CodeScaleR** | 2,896 | FC | Python | Model-generated correctness preference pairs from [CodeScaleR](https://huggingface.co/datasets/LARK-Lab/CodeScalerPair-51K), focusing on difficult pairs where rejected passes most but not all test-cases. |
| **Pie4Perf** | 1,640 | RT | — | Execution runtime preferences from the [Pie4Perf](https://huggingface.co/datasets/iNeil77/pie4perf-Train) HQ-Train subset. |

</div>

### Totals

<div align="center">

| | Samples |
|:---|:---:|
| **Total** | **354,010** |

</div>

## Filtering Procedure

The dataset undergoes thorough cleaning and decontamination:

<div align="center">

| Step | Filter | Details |
|:---:|:---|:---|
| 1 | **Max token length** | 4096 tokens (as measured by the Themis-RM tokenizer) |
| 2 | **Code syntax filtering** | Responses with AST depth < 3 filtered out |
| 3 | **Temporal cutoff** | All GitHub commit data sourced no later than March 2019 |
| 4 | **Language filtering** | Non-English prompts discarded via GlotLID classifier |
| 5 | **Perplexity filtering** | Prompts with perplexity > 1200 discarded (KenLM model on OSCAR EN corpus) |
| 6 | **Near-deduplication** | MinHash filter (shingle size 20, similarity threshold 0.75) |
| 7 | **Benchmark decontamination** | 13-gram overlap removal against Themis-CodeRewardBench, RewardBench V1, RewardBench V2, JudgeBench, and RM-Bench |

</div>

## Training Usage

This dataset is used for the **PM stage** of Themis-RM training. During training, stochastic system prompts are assigned:

<div align="center">

| Probability | System Prompt Strategy |
|:---:|:---|
| 15% | No system prompt |
| 60% | Aspect-specific prompt (Helpfulness + Harmlessness + target criterion) |
| 25% | Full multi-criteria prompt (all 5 code criteria) |

</div>

Training uses the **Bradley-Terry objective** with auxiliary conditional language modeling loss and reward magnitude regularization. One epoch, sequence length 4096, global batch size 512.

For full training details and system prompt construction, see the [Dataset](https://github.com/iNeil77/Themis/tree/main/Dataset) folder in the GitHub repository.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("project-themis/Themis-CodePreference")

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
