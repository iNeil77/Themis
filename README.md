<div align="center">

# Themis: Training Robust Multilingual Code Reward Models for Flexible Multi-Criteria Scoring

[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-Themis--RM-yellow)](https://huggingface.co/collections/project-themis/themis-reward-model-collection)
[![Datasets & Benchmarks](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets%20%26%20Benchmarks-Themis-blue)](https://huggingface.co/collections/project-themis/themis-preference-datasets-and-benchmarks)
[![Docker](https://img.shields.io/badge/Docker-ineil77%2Fthemis-2496ED?logo=docker)](https://hub.docker.com/repository/docker/ineil77/themis/general)

</div>

> **Abstract:**
>
> Reward models (RMs) have become an indispensable fixture of the language model (LM) post-training playbook, enabling policy alignment and test-time scaling. Research on the application of RMs in code generation, however, has been comparatively sparse, with existing work largely focusing on execution feedback. This choice constrains post-training to optimizing functional correctness over self-contained executable code. In this work, we examine the training and evaluation of multilingual, multi-criteria code RMs. To this end, we first compile Themis-CodeRewardBench, a benchmark to evaluate code RMs across five preference dimensions (i.e., criteria) and eight programming languages, on which we profile 50+ code, math, and general-purpose RMs. Observing the limited proficiency of current RMs beyond scoring for functional correctness, we develop Themis-CodePreference, the largest open-source collection of code preferences to date (more than 350k preference pairs), and use it to train Themis-RM, a suite of multilingual code reward models for flexible multi-criteria scoring, ranging in size from 600M to 32B parameters. Our experiments and ablations demonstrate positive scaling trends, strong cross-lingual transfer when training on diverse preferences, and the importance of multi-criteria training for reliable code reward modeling.
>

Themis reward models are trained using the Bradley-Terry preference framework with a multi-stage data pipeline that mines, filters, scores, and assembles high-quality code preference pairs from open-source repositories. The models are evaluated on Code RewardBench (CRB), a benchmark of 8,866 preference pairs spanning 5 quality aspects and 8 programming languages.

## Repository Structure

```
Themis/
├── Dataset/            Pipeline for constructing the preference training dataset
│   ├── Commit_Mining_SQL/    BigQuery SQL for extracting single-file commits
│   ├── Repos/                Curated per-language repository allowlists (~348k repos)
│   ├── Utils/                Content retrieval, deduplication, and filtering tools
│   ├── Commit_Mining_Terms/  Aspect-specific term lists for commit classification
│   └── Prompts/              Jinja2 templates, LLM judge driver, system prompt mapper
│
├── Training/           Distributed reward model training on multi-node GPU clusters
│   ├── train_reward_model.py     Training script (FSDP2 + Liger fused kernels)
│   ├── launch_reward_training.sh Per-node launch script (container + accelerate)
│   ├── Themis.Dockerfile         Container build (PyTorch + NCCL)
│   └── fsdp2_config.yaml        Accelerate / FSDP2 configuration
│
└── Evaluation/         Benchmarking suite for 51 reward models on Code RewardBench
    ├── Evaluation_Scripts/   20 per-architecture evaluation scripts
    └── Evaluation_Runs/      Pre-computed results (results.json + scores.parquet)
```

## Pipeline Overview

The end-to-end pipeline has three phases: dataset construction, model training, and evaluation.

```
                          DATASET CONSTRUCTION
                          ────────────────────
  BigQuery (github_repos)
      │
      ▼
  ┌─────────────────────┐   ┌───────────────────┐   ┌──────────────────┐
  │ 1. Commit Mining    │──▶│ 2. Repo Filtering │──▶│ 3. Ext Filtering │
  │    (SQL)            │   │    (allowlists)   │   │    (lang → ext)  │
  └─────────────────────┘   └───────────────────┘   └──────────────────┘
                                                            │
      ┌─────────────────────────────────────────────────────┘
      ▼
  ┌──────────────────────┐   ┌──────────────────┐   ┌──────────────────┐
  │ 4. Content Retrieval │──▶│ 5. Deduplication │──▶│ 6. Aspect Filter │
  │    (git fetch)       │   │    (MinHash LSH) │   │    (ModernBERT)  │
  └──────────────────────┘   └──────────────────┘   └──────────────────┘
                                                            │
      ┌─────────────────────────────────────────────────────┘
      ▼
  ┌──────────────────────┐   ┌──────────────────┐   ┌──────────────────┐
  │ 7. LLM Scoring &     │-─▶│ 8. LLM-as-a-Judge│──▶│ 9. Training Data │
  │    Instruction Synth │   │    (A/B voting)  │   │    Assembly      │
  └──────────────────────┘   └──────────────────┘   └──────────────────┘
                                                            │
                          MODEL TRAINING                    │
                          ──────────────                    │
      ┌─────────────────────────────────────────────────────┘
      ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Bradley-Terry preference training with FSDP2 on multi-node GPUs   │
  │ (BT loss + LM regularisation + magnitude penalty, Liger kernels)  │
  └───────────────────────────────────┬───────────────────────────────┘
                                      │
                          EVALUATION  │
                          ──────────  │
      ┌───────────────────────────────┘
      ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │ Code RewardBench: 8,866 pairs × 5 aspects × 8 languages           │
  │ Evaluated across scalar, MoE, and generative RM architectures     │
  └───────────────────────────────────────────────────────────────────┘
```

## Results

Themis-RM models achieve best-in-class accuracy on [Themis-CodeRewardBench](https://huggingface.co/datasets/project-themis/Themis-CodeRewardBench), a code-specific reward model benchmark, while also matching or exceeding much larger models on established general-domain benchmarks (RewardBench V1, RewardBench V2, JudgeBench). Models are grouped by parameter class; **bold** marks the best in each group.

| Model | [Themis-CodeRewardBench](https://huggingface.co/datasets/project-themis/Themis-CodeRewardBench) | [RewardBench V1](https://huggingface.co/datasets/allenai/reward-bench) | [RewardBench V2](https://huggingface.co/datasets/allenai/reward-bench-v2) | [JudgeBench](https://huggingface.co/datasets/ScalerLab/JudgeBench) |
|---|---|---|---|---|
| | | | | |
| **32B - 72B Class** | | | | |
| [WorldPM-72B](https://huggingface.co/Qwen/WorldPM-72B-RLHFLow) | 76.96 | 90.88 | 67.92 | 55.21 |
| [Athene-RM-70B](https://huggingface.co/Nexusflow/Athene-RM-70B) | 78.39 | 91.22 | 68.76 | 63.45 |
| [Nemotron-70B-Reward](https://huggingface.co/nvidia/Llama-3.3-Nemotron-70B-Reward) | 81.19 | 93.88 | 70.49 | **73.47** |
| **[Themis-RM-32B](https://huggingface.co/project-themis/Themis-RM-32B)** | **91.82** | **94.89** | **72.34** | 71.65 |
| [AceCodeRM-32B](https://huggingface.co/TIGER-Lab/AceCodeRM-32B) | 62.95 | 23.58 | 67.98 | 66.77 |
| | | | | |
| **7B – 14B Class** | | | | |
| **[Themis-RM-14B](https://huggingface.co/project-themis/Themis-RM-14B)** | **91.19** | 94.11 | 71.44 | **70.85** |
| **[Themis-RM-8B](https://huggingface.co/project-themis/Themis-RM-8B)** | 89.78 | 93.69 | 65.87 | 69.97 |
| [Athene-RM-8B](https://huggingface.co/Nexusflow/Athene-RM-8B) | 76.58 | 87.48 | 62.96 | 61.12 |
| [CodeScaler-8B](https://huggingface.co/LARK-Lab/CodeScaler-8B) | 79.12 | 94.66 | 76.51 | 70.05 |
| [Skywork-Reward-V2-8B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B) | 79.97 | **94.76** | **76.93** | 67.90 |
| [AceCodeRM-7B](https://huggingface.co/TIGER-Lab/AceCodeRM-7B) | 71.11 | 22.74 | 63.16 | 61.09 |
| | | | | |
| **0.6B - 4B Class** | | | | |
| **[Themis-RM-4B](https://huggingface.co/project-themis/Themis-RM-4B)** | **88.39** | 92.46 | 63.81 | 68.02 |
| [CodeScaler-4B](https://huggingface.co/LARK-Lab/CodeScaler-4B) | 77.97 | **94.32** | **75.13** | **68.44** |
| [Skywork-Reward-V2-4B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-4B) | 79.27 | 94.06 | 74.26 | 65.43 |
| **[Themis-RM-1.7B](https://huggingface.co/project-themis/Themis-RM-1.7B)** | 83.04 | 89.17 | 56.22 | 63.29 |
| [CodeScaler-1.7B](https://huggingface.co/LARK-Lab/CodeScaler-1.7B) | 73.75 | 91.13 | 68.44 | 66.17 |
| [Skywork-Reward-V2-1.7B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-1.7B) | 75.60 | 91.64 | 67.71 | 66.48 |
| **[Themis-RM-0.6B](https://huggingface.co/project-themis/Themis-RM-0.6B)** | 79.26 | 83.41 | 49.61 | 63.84 |
| [Skywork-Reward-V2-0.6B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B) | 72.77 | 86.32 | 60.83 | 63.65 |

## Datasets

All datasets are available on HuggingFace:

| Dataset | Description | Samples |
|---|---|---|
| [Themis-CodeRewardBench](https://huggingface.co/datasets/project-themis/Themis-CodeRewardBench) | Code RM evaluation benchmark: 5 quality dimensions, 8 languages, 19 source subsets | 8,866 |
| [Themis-CodePreference](https://huggingface.co/datasets/project-themis/Themis-CodePreference) | Training data for the PM stage: code preferences across 5 criteria and 8 languages | 354,010 |
| [Themis-GeneralPreference](https://huggingface.co/datasets/project-themis/Themis-GeneralPreference) | Training data for the PT stage: general-domain and code retrieval preferences | 110,598 |
| [Themis-Git-Commits-Merged](https://huggingface.co/datasets/project-themis/git-commits-merged) | Single-file commits from merged PRs across 24 languages (intermediate, pre-classification) | ~8M |
| [Themis-Git-Commits](https://huggingface.co/datasets/project-themis/git-commits) | Raw mined single-file commits from permissively licensed repos (full unfiltered pool) | ~28M |

## Phase 1: Dataset Construction

The dataset pipeline mines single-file commits from GitHub's BigQuery public dataset using a modified version of the commit mining pipeline from [OctoPack](https://arxiv.org/abs/2308.07124) ([CommitPack](https://huggingface.co/datasets/bigcode/commitpack)), filters them through repository reputation and language allowlists, retrieves file contents via shallow git fetches, deduplicates with MinHash LSH, classifies commits by quality aspect using term matching and ModernBERT classifiers, scores them with LLM judges, synthesises instructions, and assembles the final preference pairs with stochastic system prompts.

The SQL query restricts to **permissively licensed** repositories only (MIT, Apache-2.0, BSD-2/3-Clause, ISC, CC0-1.0, EPL-1.0, MPL-2.0, Unlicense, AGPL-3.0, LGPL-2.1, Artistic-2.0). The BigQuery GitHub snapshot used contains commits up to **early 2022** — predating the widespread availability of LLM code generation tools — ensuring that all mined code changes represent **genuine human-authored preferences**. This raw pool is subsequently subset by time: training commit data is sourced no later than **March 2019**; benchmark commit data is scoped to **June 2019 – January 2021**, from disjoint repositories.

**9 stages** covering BigQuery extraction, repository and extension filtering, content retrieval, deduplication, aspect-based term filtering, LLM scoring and instruction synthesis, LLM-as-a-judge preference labelling, and training data assembly.

See **[Dataset/README.md](./Dataset/README.md)** for the full pipeline documentation, including per-stage scripts, arguments, invocation examples, and template reference.

### Key Components

| Component | Description |
|---|---|
| [`consolidated_query.sql`](./Dataset/Commit_Mining_SQL/consolidated_query.sql) | BigQuery SQL extracting single-file, licensed commits with non-trivial messages (modified from [OctoPack](https://arxiv.org/abs/2308.07124)) |
| [`Repos/*.json`](./Dataset/Repos/) | Curated allowlists of ~348k high-reputation repos across 33 languages |
| [`retrieve_commit_contents.py`](./Dataset/Utils/retrieve_commit_contents.py) | Multi-threaded shallow git fetch of pre/post commit file contents |
| [`minHash_dedupe_local.py`](./Dataset/Utils/minHash_dedupe_local.py) | MinHash LSH deduplication adapted from BigCode |
| [`Prompts/templates/*.j2`](./Dataset/Prompts/templates/) | 6 Jinja2 templates for LLM scoring, instruction synthesis, and judging |
| [`gen_llm.py`](./Dataset/Prompts/gen_llm.py) | vLLM driver for multi-sample LLM-as-a-judge preference labelling |
| [`system_prompt_mapper.py`](./Dataset/Prompts/system_prompt_mapper.py) | Stochastic system-prompt assignment for training data |

## Phase 2: Training

Themis-RM models are trained in two stages using the Bradley-Terry preference framework on multi-node GPU clusters. The training script supports FSDP2 distributed training, Liger fused Triton kernels for efficient linear cross-entropy computation, and an optional LM regularisation loss.

1. **Preference Model Pre-Training (PT):** Trains on [Themis-GeneralPreference](https://huggingface.co/datasets/project-themis/Themis-GeneralPreference) (110k+ general-domain and code retrieval preferences) for 2 epochs to instill common human-inspired notions of preference evaluation such as relevance, helpfulness, and harmlessness.
2. **Preference Modeling (PM):** Trains on [Themis-CodePreference](https://huggingface.co/datasets/project-themis/Themis-CodePreference) (350k+ code preference pairs across 5 quality dimensions and 8 programming languages) for 1 epoch to specialize on multi-criteria code scoring.

See **[Training/README.md](./Training/README.md)** for the full training documentation, including container setup, cluster configuration, environment variables, FSDP2 config, all training arguments, launch commands, monitoring, checkpointing, and troubleshooting.

### Training Hyperparameters

| Attribute | PT Stage | PM Stage |
|---|---|---|
| Training Dataset | Themis-GeneralPreference | Themis-CodePreference |
| Peak Learning Rate | 2e-5 | 1e-5 |
| Terminal Learning Rate | 1e-5 | 5e-7 |
| LM Regularisation Coefficient (λ) | 0.4 | 0.25 |
| Reward Magnitude Coefficient (μ) | 0.01 | 0.001 |
| Scheduler | Cosine (5% warmup) | Cosine (5% warmup) |
| Weight Decay | 0.1 | 0.1 |
| Gradient Clipping | 2.0 | 1.5 |
| Global Batch Size | 1024 | 512 |
| Sequence Length | 2560 | 4096 |
| Training Epochs | 2 | 1 |
| Optimizer | AdamW-Fused (β = {0.9, 0.95}) | AdamW-Fused (β = {0.9, 0.95}) |
| Precision | bfloat16 | bfloat16 |

### Key Components

| Component | Description |
|---|---|
| [`train_reward_model.py`](./Training/train_reward_model.py) | Training script: `RewardModelWithLMHead` backbone + reward head, `PairCollator`, BT + LM + magnitude loss |
| [`launch_reward_training.sh`](./Training/launch_reward_training.sh) | Per-node launch script: container runtime with accelerate and CUDA/NCCL env vars |
| [`Themis.Dockerfile`](./Training/Themis.Dockerfile) | Container build: PyTorch NGC base + high-speed networking + Open MPI |
| [`fsdp2_config.yaml`](./Training/fsdp2_config.yaml) | Accelerate / FSDP2 distributed training configuration |

## Phase 3: Evaluation

The evaluation suite benchmarks reward models on Code RewardBench (CRB) — 8,866 preference pairs across 5 quality aspects (Functional Correctness, Runtime Efficiency, Memory Efficiency, Security Hardness, Readability & Maintainability), 8 programming languages, and 19 source subsets. The suite supports 4 architecture categories across 20 scripts and includes pre-computed results for 51 models.

See **[Evaluation/README.md](./Evaluation/README.md)** for the full evaluation documentation, including the benchmark dataset schema, evaluation protocol, per-script invocation examples, output format, and the complete results table.

### Architecture Categories

| Category | Scripts | Description |
|---|---|---|
| Standard scalar | `coderewardbench-seqcls.py` | `AutoModelForSequenceClassification` — works for most HuggingFace RMs |
| Custom scalar | 13 scripts (armo, qrm, athene, inform, acecode, ldl, grm, internlm, nemotron, starling, eurus, ultra, automodel) | Custom model classes with MoE gating, value heads, quantile regression, etc. |
| Generative | 4 scripts (cerm, nemotron-genrm, lmunit, r3) | vLLM-based text generation with score parsing |
| Reranking | `rerank_eval.py` | Reward variance, Hits@K, and Spearman correlation on code completions |

## Key Experimental Findings

Our experiments (detailed in the [paper](https://arxiv.org/abs/xxxx.xxxxx)) investigate four research questions:

- **RQ1 — Multi-criteria code scoring:** Existing RMs are largely unusable for scoring code along non-functional axes (efficiency, security), often degenerating to random scoring. Themis-RM-0.6B outscores multiple >100x larger general-purpose RMs, while Themis-RM-32B sets a clear state-of-the-art across all criteria. Scalar reward modeling is well-suited to reference-free code evaluation — the most competitive existing RMs are all scalar RMs.

- **RQ2 — Minimizing cross-criteria interference:** Both the PT phase and auxiliary training losses (LM regularisation, magnitude penalty) improve performance across the board. Criteria-conditioned system prompts effectively disentangle multi-dimensional preferences, eliminating the need for criteria-specific modules, ensembling, or model merging. Training on functional correctness preferences transfers well to non-functional criteria, and positive transfer between all quality criteria means no single-criterion model matches the full multi-criteria Themis-RM.

- **RQ3 — Cross-lingual transfer:** Training on all eight programming languages yields the best performance, suggesting net-positive cross-lingual transfer. Python RMs transfer better to dynamically typed languages; Java RMs transfer better to statically typed ones. Training on diverse multi-criteria preferences leads to stable multilingual reward modeling with small performance differences across languages.

- **RQ4 — Downstream robustness:** Themis-RM achieves state-of-the-art adversarial robustness on judge-hacking perturbations and matches the best RMs specialized for correctness in listwise re-ranking of code contest solutions — a task proven to predict downstream post-training utility. Scaling trends in downstream robustness are even stronger than in pairwise accuracy.

## Related Work

**[Distributed Training Tutorial](https://github.com/iNeil77/AWS_DistTraining_Tutorial)** — A companion tutorial by us that walks through multi-node distributed training of scalar reward models on cloud GPU clusters. Covers cluster provisioning, high-speed networking, container management, and FSDP-based training. Useful as a standalone guide for anyone looking to reproduce the Themis training setup or adapt it to their own reward modelling workloads. Follows a simplified recipe that leverages the Axolotl framework for training reward models with the Bradley-Terry loss.

## Citation

```bibtex
@article{themis2025,
  title={Themis: Training Robust Multilingual Code Reward Models for Flexible Multi-Criteria Scoring},
  author={Paul, Indraneil and Gurevych, Iryna and Glava\v{s}, Goran},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

## License

Apache 2.0 — see [LICENSE](./LICENSE).
