---
tags:
- code
- github
- commits
- multilingual
license: apache-2.0
task_categories:
- text-generation
language:
- code
size_categories:
- 1M<n<10M
---

<div align="center">

# Themis-Git-Commits

[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-Themis--RM-yellow)](https://huggingface.co/collections/project-themis/themis-reward-model-collection)
[![Datasets & Benchmarks](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets%20%26%20Benchmarks-Themis-blue)](https://huggingface.co/collections/project-themis/themis-preference-datasets-and-benchmarks)
[![GitHub](https://img.shields.io/badge/GitHub-Themis-181717?logo=github)](https://github.com/iNeil77/Themis)

</div>

## Overview

**Themis-Git-Commits** is a large-scale dataset of single-file code commits mined from **permissively licensed** GitHub repositories via the [BigQuery GitHub public dataset](https://console.cloud.google.com/marketplace/product/github/github-repos). The SQL query restricts to repositories under permissive open-source licenses only (MIT, Apache-2.0, BSD-2/3-Clause, ISC, CC0-1.0, EPL-1.0, MPL-2.0, Unlicense, AGPL-3.0, LGPL-2.1, Artistic-2.0). The BigQuery snapshot used contains commits up to **early 2022** — predating the widespread availability of LLM code generation tools — ensuring that all code changes in the dataset represent **genuine human-authored preferences**.

This is the **raw commit dataset** — prior to merging with pull request data to subset only for merged commits. It serves as the foundational data source for the commit-based preference pairs in [Themis-CodePreference](https://huggingface.co/datasets/project-themis/Themis-CodePreference), which is used to train the [Themis-RM](https://huggingface.co/collections/project-themis/themis-reward-model-collection) suite of multilingual code reward models.

Each row represents a single commit that changes exactly one file in a repository with a permissive open-source license. The dataset includes the commit metadata (SHA, message, timestamp, license) along with the pre-commit and post-commit file contents, enabling downstream construction of code-change preference pairs across multiple quality dimensions.

## Collection Pipeline

The commit mining pipeline is described in detail in the [Themis paper](https://arxiv.org/abs/xxxx.xxxxx) and the [Dataset](https://github.com/iNeil77/Themis/tree/main/Dataset) folder in the GitHub repository. The BigQuery SQL query and scraping infrastructure are modified from the [OctoPack](https://arxiv.org/abs/2308.07124) pipeline ([CommitPack](https://huggingface.co/datasets/bigcode/commitpack)); the subsequent filtering, classification, and preference construction stages are original to Themis. At a high level:

1. **BigQuery Mining** — A [GoogleSQL query](https://github.com/iNeil77/Themis/blob/main/Dataset/Commit_Mining_SQL/consolidated_query.sql) (modified from [OctoPack](https://arxiv.org/abs/2308.07124)) extracts single-file commits from `bigquery-public-data.github_repos`, filtering for permissive licenses, target programming languages, and non-trivial commit messages.

2. **Repository Reputation Filtering** — Commits are subset to those originating from [curated high-reputation repositories](https://github.com/iNeil77/Themis/tree/main/Dataset/Repos) (15+ GitHub stars, 5+ contributors, 10+ issues).

3. **Content Retrieval** — The pre-commit (`old_contents`) and post-commit (`new_contents`) file contents are fetched from GitHub via shallow git fetches using [retrieve_commit_contents.py](https://github.com/iNeil77/Themis/blob/main/Dataset/Utils/retrieve_commit_contents.py).

4. **MinHash Deduplication** — Near-duplicate content is removed using [MinHash LSH deduplication](https://github.com/iNeil77/Themis/blob/main/Dataset/Utils/minHash_dedupe_local.py) (shingle size 5, 256 permutations, Jaccard threshold 0.7).

## Downstream Processing (Not in This Dataset)

The steps below are applied downstream and are **not** reflected in this raw dataset:

- **Extension Filtering** — Commits are filtered so the changed file's extension matches a target programming language. Applied in [Themis-Git-Commits-Merged](https://huggingface.co/datasets/project-themis/git-commits-merged).
- **Pull Request Cross-Referencing** — Commits are cross-referenced with [GHTorrent](https://ghtorrent.org/) pull request data (through end of 2021) to retain only non-reverted commits that are part of successfully merged pull requests, ensuring implicit human validation. Applied in [Themis-Git-Commits-Merged](https://huggingface.co/datasets/project-themis/git-commits-merged).
- **Temporal Subsetting** — For training data ([Themis-CodePreference](https://huggingface.co/datasets/project-themis/Themis-CodePreference)), only commits pushed before **March 2019** are retained. For benchmark data ([Themis-CodeRewardBench](https://huggingface.co/datasets/project-themis/Themis-CodeRewardBench)), commits are scoped to **June 2019 – January 2021** from disjoint repositories.
- **Aspect Classification** — Commits are assigned to quality dimensions (Functional Correctness, Runtime Efficiency, Memory Efficiency, Security Hardness, Readability & Maintainability) using criteria-specialized [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) commit classifiers, trained on seed positives retrieved via [curated term lists](https://github.com/iNeil77/Themis/tree/main/Dataset/Commit_Mining_Terms).
- **LLM Scoring & Instruction Synthesis** — Frontier LMs validate preference strength and generate realistic inverse instructions.
- **LLM-as-a-Judge Preference Labelling** — Multi-sample voting with frontier LMs produces consensus preference labels.

## Dataset Schema

<div align="center">

| Column | Type | Description |
|:---|:---:|:---|
| `commit` | string | Git commit SHA |
| `subject` | string | First line of the commit message |
| `message` | string | Full commit message body |
| `repos` | string | Comma-separated list of repository names containing this commit |
| `file_path` | string | Path of the changed file |
| `license` | string | SPDX license identifier of the source repository |
| `unix_time` | int64 | Committer timestamp (seconds since epoch) |
| `new_contents` | string | File contents after the commit (post-commit) |
| `old_contents` | string | File contents before the commit (pre-commit) |

</div>

## Filters Applied During Mining

<div align="center">

| Filter | Purpose |
|:---|:---|
| **License allowlist** | MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, CC0-1.0, EPL-1.0, MPL-2.0, Unlicense, AGPL-3.0, LGPL-2.1, Artistic-2.0 |
| **Language allowlist** | Python, Java, JavaScript, C, C#, C++, TypeScript, Go, Ruby |
| **Message length** | 10 < length < 15,000 characters |
| **Message blocklist** | ~50 low-signal messages excluded (e.g., "initial commit", "wip", "yolo") |
| **Pattern exclusion** | Merge commits and CI push messages filtered out |
| **Same-path constraint** | `old_path = new_path` — file was modified in place, not renamed or moved |
| **Single-file constraint** | Commit touches exactly one file |
| **Content retrieval** | Both pre-commit and post-commit file contents successfully fetched |
| **Near-deduplication** | MinHash LSH with Jaccard threshold 0.7 |

</div>

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("project-themis/git-commits")

# Inspect a sample
sample = dataset["train"][0]
print(f"Commit: {sample['commit']}")
print(f"Subject: {sample['subject']}")
print(f"License: {sample['license']}")
print(f"File: {sample['file_path']}")
print(f"Old contents length: {len(sample['old_contents'])}")
print(f"New contents length: {len(sample['new_contents'])}")
```

## License

This dataset is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). The source commits are drawn exclusively from repositories with permissive open-source licenses (see filter table above).

## Citation

```bibtex
@article{themis2025,
  title={Themis: Training Robust Multilingual Code Reward Models for Flexible Multi-Criteria Scoring},
  author={Paul, Indraneil and Gurevych, Iryna and Glava\v{s}, Goran},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```
