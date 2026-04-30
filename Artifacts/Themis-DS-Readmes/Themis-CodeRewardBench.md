---
dataset_info:
- config_name: Full
  features:
  - name: prompt
    dtype: large_string
  - name: chosen
    dtype: large_string
  - name: rejected
    dtype: large_string
  - name: language
    dtype: string
  - name: aspect
    dtype: string
  - name: subset
    dtype: string
  - name: id
    dtype: string
  splits:
  - name: Full
    num_bytes: 59924734
    num_examples: 8866
  download_size: 25573996
  dataset_size: 59924734
- config_name: Stratified_Aspect
  features:
  - name: prompt
    dtype: large_string
  - name: chosen
    dtype: large_string
  - name: rejected
    dtype: large_string
  - name: language
    dtype: string
  - name: aspect
    dtype: string
  - name: subset
    dtype: string
  - name: id
    dtype: string
  splits:
  - name: Memory_Efficiency
    num_bytes: 1953332.7459959395
    num_examples: 289
  - name: Runtime_Efficiency
    num_bytes: 8847448.320099255
    num_examples: 1309
  - name: Security_Hardness
    num_bytes: 6698106.405819986
    num_examples: 991
  - name: Functional_Correctness
    num_bytes: 32294200.208887886
    num_examples: 4778
  - name: Readability_Maintainability
    num_bytes: 10131646.319196932
    num_examples: 1499
  download_size: 84274714
  dataset_size: 59924733.99999999
- config_name: Stratified_Language
  features:
  - name: prompt
    dtype: large_string
  - name: chosen
    dtype: large_string
  - name: rejected
    dtype: large_string
  - name: language
    dtype: string
  - name: aspect
    dtype: string
  - name: subset
    dtype: string
  - name: id
    dtype: string
  splits:
  - name: Ruby
    num_bytes: 6252016.574554478
    num_examples: 925
  - name: JavaScript
    num_bytes: 7326687.531694112
    num_examples: 1084
  - name: Java
    num_bytes: 9915360.340401534
    num_examples: 1467
  - name: Cpp
    num_bytes: 9942396.087750958
    num_examples: 1471
  - name: CSharp
    num_bytes: 1270680.125422964
    num_examples: 188
  - name: C
    num_bytes: 2967173.271599368
    num_examples: 439
  - name: Python
    num_bytes: 17343431.92465599
    num_examples: 2566
  - name: Go
    num_bytes: 4906988.143920596
    num_examples: 726
  download_size: 90792044
  dataset_size: 59924733.99999999
configs:
- config_name: Full
  data_files:
  - split: Full
    path: Full/Full-*
  default: true
- config_name: Stratified_Aspect
  data_files:
  - split: Memory_Efficiency
    path: Stratified_Aspect/Memory_Efficiency-*
  - split: Readability_Maintainability
    path: Stratified_Aspect/Readability_Maintainability-*
  - split: Security_Hardness
    path: Stratified_Aspect/Security_Hardness-*
  - split: Runtime_Efficiency
    path: Stratified_Aspect/Runtime_Efficiency-*
  - split: Functional_Correctness
    path: Stratified_Aspect/Functional_Correctness-*
- config_name: Stratified_Language
  data_files:
  - split: Java
    path: Stratified_Language/Java-*
  - split: C
    path: Stratified_Language/C-*
  - split: Python
    path: Stratified_Language/Python-*
  - split: Ruby
    path: Stratified_Language/Ruby-*
  - split: Cpp
    path: Stratified_Language/Cpp-*
  - split: Go
    path: Stratified_Language/Go-*
  - split: JavaScript
    path: Stratified_Language/JavaScript-*
  - split: CSharp
    path: Stratified_Language/CSharp-*
tags:
- code
- reward-model
- benchmark
- multilingual
license: apache-2.0
task_categories:
- text-ranking
- text-classification
size_categories:
- 1K<n<10K
---

<div align="center">

# Themis-CodeRewardBench

[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-Themis--RM-yellow)](https://huggingface.co/collections/project-themis/themis-reward-model-collection)
[![Datasets & Benchmarks](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets%20%26%20Benchmarks-Themis-blue)](https://huggingface.co/collections/project-themis/themis-preference-datasets-and-benchmarks)
[![GitHub](https://img.shields.io/badge/GitHub-Themis-181717?logo=github)](https://github.com/iNeil77/Themis)

</div>

## Overview

**Themis-CodeRewardBench** is a code-specific reward model evaluation benchmark comprising ~8.9k diverse code preference pairs across eight programming languages and five quality scoring dimensions. It is part of the [Themis](https://github.com/iNeil77/Themis) project and evaluates code reward models on five code quality dimensions — **Functional Correctness (FC)**, **Execution Efficiency (EE)**, **Memory Efficiency (ME)**, **Readability & Maintainability (R&M)**, and **Security Hardness (SH)** — across eight programming languages: C, C#, C++, Go, Java, JavaScript, Python, and Ruby.

The benchmark uses **preference accuracy** as the evaluation metric. It draws from 13 distinct pre-existing and newly constructed code preference datasets, spanning human-written, LLM-generated, and mixed-provenance prompts and responses. It introduces a largely novel distribution of code preferences, for code of increased complexity, compared to the code subsets in existing RM benchmarks.

**Key differentiators:**

- Evaluates across 5 quality dimensions, not just functional correctness
- Covers 8 programming languages, not just Python
- Includes human-written code from real commits, not only contest/synthetic code
- Introduces a novel distribution of code preferences with increased code complexity compared to existing RM benchmarks

## Dataset Configurations

The dataset is published with three configurations, each offering a different view of the same 8,866 samples:

<div align="center">

| Config | Splits | Description |
|:---|:---|:---|
| **`Full`** (default) | `Full` | All 8,866 samples in a single split |
| **`Stratified_Aspect`** | `Functional_Correctness`, `Runtime_Efficiency`, `Memory_Efficiency`, `Security_Hardness`, `Readability_Maintainability` | Samples partitioned by quality dimension |
| **`Stratified_Language`** | `C`, `CSharp`, `Cpp`, `Go`, `Java`, `JavaScript`, `Python`, `Ruby` | Samples partitioned by programming language |

</div>

## Dataset Schema

<div align="center">

| Column | Type | Description |
|:---|:---:|:---|
| `id` | string | Unique sample identifier |
| `prompt` | string | User query / coding task |
| `chosen` | string | Preferred response (ground truth) |
| `rejected` | string | Dispreferred response |
| `aspect` | string | Quality axis: `Functional_Correctness`, `Memory_Efficiency`, `Readability_Maintainability`, `Runtime_Efficiency`, `Security_Hardness` |
| `language` | string | Programming language: C, CSharp, Cpp, Go, Java, JavaScript, Python, Ruby |
| `subset` | string | Source dataset identifier |

</div>

## Aspect Distribution

<div align="center">

| Aspect | Pairs |
|:---|:---:|
| Functional Correctness | 4,778 |
| Readability & Maintainability | 1,499 |
| Runtime Efficiency | 1,309 |
| Security Hardness | 991 |
| Memory Efficiency | 289 |
| **Total** | **8,866** |

</div>

## Language Distribution

<div align="center">

| Language | Pairs |
|:---|:---:|
| Python | 2,566 |
| C++ | 1,471 |
| Java | 1,467 |
| JavaScript | 1,084 |
| Ruby | 925 |
| Go | 726 |
| C | 439 |
| C# | 188 |

</div>

## Benchmark Composition

The benchmark is composed of 13 constituent datasets grouped by quality dimension. Each cell shows the language and the number of preference pairs contributed. "Prompts" and "Responses" columns indicate the provenance of the respective text.

### Functional Correctness (FC)

<div align="center">

| Dataset | Languages (pair count) | Prompts | Responses |
|:---|:---|:---:|:---:|
| **Commit Preference Correctness** | C(11), C++(25), C#(42), Go(59), Java(134), JS(241), Python(216), Ruby(97) | LLM Generated | Human Written |
| **HumanEvalPack** (Muennighoff et al., 2024) | C++(147), Go(117), Java(132), JS(113), Python(119) | Human Written | Human Written |
| **MBPPPlusFix-Hard** (Prasad et al., 2025) | Python(37) | Human Written | Mixed |
| **MDEval** (Liu et al., 2024) | C(12), C++(11), Go(24), JS(30), Python(22), Ruby(35) | Human Written | Human Written |
| **DebugEval** (Yang et al., 2025) | C++(211), Java(194), Python(319) | Human Written | Human Written |
| **RunBugRun-V1** (Prenner et al., 2023) | C(279), C++(504), Go(352), Java(453), JS(164), Python(305), Ruby(376) | Human Written | Human Written |

</div>

### Execution Efficiency (EE)

<div align="center">

| Dataset | Languages (pair count) | Prompts | Responses |
|:---|:---|:---:|:---:|
| **Commit Preference Runtime** | C(4), C++(6), C#(21), Go(15), Java(55), JS(39), Python(56), Ruby(42) | LLM Generated | Human Written |
| **Pie4Perf** (Shypula et al., 2024) | C++(460) | Human Written | Human Written |
| **ECCO** (Waghjale et al., 2024) | Python(399) | Human Written | Human Written |
| **EvalPerf** (Liu et al., 2024) | Python(212) | Human Written | LLM Generated |

</div>

### Memory Efficiency (ME)

<div align="center">

| Dataset | Languages (pair count) | Prompts | Responses |
|:---|:---|:---:|:---:|
| **Commit Preference Memory** | C(73), C++(43), C#(8), Go(12), Java(52), JS(28), Python(26), Ruby(10) | LLM Generated | Human Written |
| **NoFunEval Memory** (Singhal et al., 2024) | C(2), Java(35) | LLM Generated | Human Written |

</div>

### Readability & Maintainability (R&M)

<div align="center">

| Dataset | Languages (pair count) | Prompts | Responses |
|:---|:---|:---:|:---:|
| **Commit Preference CodeStyle** | C(13), C++(24), C#(80), Go(101), Java(257), JS(325), Python(365), Ruby(206) | LLM Generated | Human Written |
| **NoFunEval Maintain** (Singhal et al., 2024) | Python(128) | LLM Generated | Human Written |

</div>

### Security Hardness (SH)

<div align="center">

| Dataset | Languages (pair count) | Prompts | Responses |
|:---|:---|:---:|:---:|
| **Commit Preference Security** | C(31), C++(42), C#(37), Go(46), Java(143), JS(144), Python(172), Ruby(206) | LLM Generated | Human Written |
| **CodePrefBench Security** (Liu et al., 2024) | Python(173) | LLM Generated | Mixed |
| **Vul4J** (Bui et al., 2022) | Java(8) | LLM Generated | Human Written |
| **SecBench** (Jing et al., 2024) | C(2), C++(1), Java(4), Python(2), Ruby(5) | LLM Generated | Human Written |
| **NoFunEval Security** (Singhal et al., 2024) | C(12), Python(15) | LLM Generated | LLM Generated |

</div>

## Commit Preference Construction

The "Commit Preference" subsets (Correctness, Runtime, Memory, CodeStyle, Security) are **newly constructed** for this benchmark using the pipeline described in the [Themis paper](https://arxiv.org/abs/xxxx.xxxxx). The commit mining SQL query and scraping infrastructure are modified from the [OctoPack](https://arxiv.org/abs/2308.07124) pipeline ([CommitPack](https://huggingface.co/datasets/bigcode/commitpack)); the subsequent filtering, classification, consensus voting, and instruction synthesis stages are original to Themis. The query restricts to **permissively licensed** repositories only, and the BigQuery snapshot contains commits up to **early 2022** — predating widespread LLM code generation — guaranteeing that all mined code represents genuine human-authored preferences.

1. **Single-file commit mining** — Commits mined from **permissively licensed** repositories in the [BigQuery GitHub public dataset](https://console.cloud.google.com/marketplace/product/github/github-repos) using a modified [OctoPack](https://arxiv.org/abs/2308.07124) query. The raw commits are published as [Themis-Git-Commits](https://huggingface.co/datasets/project-themis/git-commits).
2. **Repository reputation filtering** — Filtered to merged pull requests from reputable repositories (15+ stars, 5+ contributors, 10+ issues) via [GHTorrent](https://ghtorrent.org/). The merged subset is published as [Themis-Git-Commits-Merged](https://huggingface.co/datasets/project-themis/git-commits-merged).
3. **Temporal scoping** — Commits authored between **June 2019** and **January 2021**, from disjoint repositories vs. training data.
4. **Aspect classification** — Classified using criteria-specialized [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) commit classifiers, trained on search-term-recalled samples.
5. **Multi-LLM consensus voting** — Multiple frontier open-source LMs verify that each commit reflects a single clear intent, weeding out multi-purpose changes.
6. **Inverse instruction synthesis** — Realistic user instructions are synthetically generated across multiple problem styles (e.g., contest problem, StackOverflow question, search query).

For the full pipeline, see the [Dataset](https://github.com/iNeil77/Themis/tree/main/Dataset) folder in the GitHub repository.

## Evaluation

**Preference accuracy** is the evaluation metric. For each sample, the reward model scores both the chosen and rejected responses given the prompt. The reward model is considered correct on that sample if it assigns a strictly higher score to the chosen response than to the rejected response.

Pre-computed evaluation results for 51 reward models (scalar, MoE, and generative architectures) are available in the [Evaluation](https://github.com/iNeil77/Themis/tree/main/Evaluation) folder of the GitHub repository.

## Usage

```python
from datasets import load_dataset

# Load the full benchmark (default config)
dataset = load_dataset("project-themis/Themis-CodeRewardBench")
sample = dataset["Full"][0]
print(f"Aspect: {sample['aspect']}")
print(f"Language: {sample['language']}")
print(f"Subset: {sample['subset']}")

# Load a single aspect split
fc = load_dataset("project-themis/Themis-CodeRewardBench", "Stratified_Aspect", split="Functional_Correctness")
print(f"FC samples: {len(fc)}")

# Load a single language split
py = load_dataset("project-themis/Themis-CodeRewardBench", "Stratified_Language", split="Python")
print(f"Python samples: {len(py)}")
```

## License

This dataset is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). The constituent datasets retain their original licenses; please refer to each source dataset for details.

## Citation

```bibtex
@article{themis2025,
  title={Themis: Training Robust Multilingual Code Reward Models for Flexible Multi-Criteria Scoring},
  author={Paul, Indraneil and Gurevych, Iryna and Glava\v{s}, Goran},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```
