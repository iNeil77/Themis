# Dataset Construction Runbook

End-to-end guide for building the Themis preference dataset from raw git commits. The pipeline mines single-file commits from BigQuery using a modified version of the commit mining pipeline from [OctoPack](https://arxiv.org/abs/2308.07124) ([CommitPack](https://huggingface.co/datasets/bigcode/commitpack)), filters them against a curated allowlist of high-reputation repositories, retrieves their contents from GitHub, deduplicates them, filters by aspect-specific terms, scores them with LLM judges, and synthesises instruction-response preference pairs.

The SQL query restricts to **permissively licensed** repositories only. The BigQuery GitHub snapshot used contains commits up to **early 2022** — predating the widespread availability of LLM code generation tools — ensuring that all mined code changes represent genuine human-authored preferences. This raw pool is subsequently subset by time for training (before March 2019) and benchmark (June 2019 – January 2021) splits.

**Published datasets on HuggingFace:**

| Dataset | Description |
|---|---|
| [Themis-Git-Commits](https://huggingface.co/datasets/project-themis/git-commits) | Raw mined single-file commits (~12M) — output of stages 1–5 |
| [Themis-Git-Commits-Merged](https://huggingface.co/datasets/project-themis/git-commits-merged) | Commits from merged PRs (~3.98M, 24 languages) — output of stage 6 (PR cross-referencing) |
| [Themis-CodePreference](https://huggingface.co/datasets/project-themis/Themis-CodePreference) | Training dataset for the PM stage (354k preference pairs) — final output |
| [Themis-GeneralPreference](https://huggingface.co/datasets/project-themis/Themis-GeneralPreference) | Training dataset for the PT stage (110k+ preference pairs) |
| [Themis-CodeRewardBench](https://huggingface.co/datasets/project-themis/Themis-CodeRewardBench) | Code RM evaluation benchmark (8,866 preference pairs) |

**Related files:**

| Directory / File | Purpose |
|---|---|
| [`Commit_Mining_SQL/consolidated_query.sql`](./Commit_Mining_SQL/consolidated_query.sql) | BigQuery SQL to extract single-file commits from `bigquery-public-data.github_repos` |
| [`Repos/`](./Repos/) | Pre-compiled per-language allowlists of high-reputation repositories (`.json`) |
| [`Utils/scrape_git_repos.py`](./Utils/scrape_git_repos.py) | Clones repos and serialises them with `yek`; used to build and extend the `Repos/` allowlists |
| [`Utils/programming_languages_ext_map.json`](./Utils/programming_languages_ext_map.json) | Mapping from language names to file extensions for extension-based filtering |
| [`Utils/retrieve_commit_contents.py`](./Utils/retrieve_commit_contents.py) | Multi-threaded scraper that shallow-fetches old/new file contents for each commit from GitHub |
| [`Utils/minHash_dedupe_local.py`](./Utils/minHash_dedupe_local.py) | MinHash LSH deduplication (adapted from BigCode) |
| [`Commit_Mining_Terms/`](./Commit_Mining_Terms/) | Minimised term lists (`.list`, one term per line) for commit message filtering, one per quality aspect |
| [`Utils/requirements_treesitter.txt`](./Utils/requirements_treesitter.txt) | Tree-sitter parser dependencies (for downstream code analysis) |
| [`Prompts/templates/`](./Prompts/templates/) | Jinja2 prompt templates for all LLM-driven pipeline stages |
| [`Prompts/gen_llm.py`](./Prompts/gen_llm.py) | vLLM-based LLM-as-a-judge driver: generates multiple comparative judgements per sample |
| [`Prompts/inference.py`](./Prompts/inference.py) | Reward model evaluation harness for the CRB benchmark |
| [`Prompts/system_prompt_mapper.py`](./Prompts/system_prompt_mapper.py) | Stochastic system-prompt constructor for training data (aspect-specific or full criteria) |

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Prerequisites](#2-prerequisites)
3. [Stage 1 — Commit Mining (BigQuery)](#3-stage-1--commit-mining-bigquery)
4. [Stage 2 — Repository Reputation Filtering](#4-stage-2--repository-reputation-filtering)
5. [Stage 3 — Extension Filtering](#5-stage-3--extension-filtering)
6. [Stage 4 — Content Retrieval](#6-stage-4--content-retrieval)
7. [Stage 5 — MinHash Deduplication](#7-stage-5--minhash-deduplication)
8. [Stage 6 — Term-Based Aspect Filtering & Commit Classification](#8-stage-6--term-based-aspect-filtering--commit-classification)
9. [Stage 7 — LLM Scoring & Instruction Synthesis](#9-stage-7--llm-scoring--instruction-synthesis)
10. [Stage 8 — LLM-as-a-Judge Preference Labelling](#10-stage-8--llm-as-a-judge-preference-labelling)
11. [Stage 9 — System Prompt Assignment & Training Data Assembly](#11-stage-9--system-prompt-assignment--training-data-assembly)
12. [Auxiliary: Building & Extending the Repo Allowlists](#12-auxiliary-building--extending-the-repo-allowlists)
13. [Auxiliary: Reward Model Evaluation](#13-auxiliary-reward-model-evaluation)

---

## 1. Pipeline Overview

```
BigQuery (github_repos)
    │
    ▼
┌───────────────────────────┐
│ Stage 1: Commit Mining    │  consolidated_query.sql
│ (single-file, licensed,   │  → exports commit metadata
│  non-trivial messages)    │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ Stage 2: Repo Reputation  │  Repos/*.json
│ Filter (keep only commits │  → subset to commits from curated
│  from high-rep repos)     │     high-reputation repositories
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ Stage 3: Extension Filter │  programming_languages_ext_map.json
│ (keep only target lang    │  → subset by changed file extension
│  extensions)              │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ Stage 4: Content Retrieval│  retrieve_commit_contents.py
│ (git fetch old + new file │  → adds old_contents, new_contents
│  from GitHub)             │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ Stage 5: Deduplication    │  minHash_dedupe_local.py
│ (MinHash LSH on file      │  → removes near-duplicate content
│  contents)                │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ Stage 6: Aspect Filtering │  Commit_Mining_Terms/*.list
│ (term seed retrieval +    │  → ModernBERT commit classifiers
│  ModernBERT classifiers)  │  → aspect-labelled commit subsets
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ Stage 7: LLM Scoring &    │  templates/commit_review*.j2
│ Instruction Synthesis     │  templates/inverse_instruction.j2
│ (rate changes 1-5,        │  templates/bug_introduction.j2
│  generate instructions)   │  templates/problem_recovery.j2
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ Stage 8: LLM-as-a-Judge   │  templates/preference_judge.j2
│ Preference Labelling      │  gen_llm.py
│ (multi-sample voting)     │  → comparative judgements (A/B/TIE)
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ Stage 9: Training Data    │  system_prompt_mapper.py
│ Assembly                  │  → final (input, chosen, rejected,
│ (system prompt assignment,│     system) preference dataset
│  tokenisation stats)      │
└───────────────────────────┘
```

---

## 2. Prerequisites

**Accounts & Access:**
- Google Cloud project with BigQuery access to `bigquery-public-data.github_repos`
- GitHub SSH key configured for `git@github.com:` fetches (rate-limited; many repos will require retries)
- OpenAI-compatible API endpoint for batch requests (used by the Jinja2 prompt templates)
- GPU instance with vLLM for local judge inference (`gen_llm.py`)
- HuggingFace account for uploading/downloading intermediate datasets

**Software:**
- Python 3.10+
- `datasets`, `torch`, `vllm`, `transformers`, `typer`, `tiktoken`, `numpy`, `scipy`, `tqdm`, `pandas`
- `yek` CLI tool (for repo serialisation in `scrape_git_repos.py`)
- Tree-sitter parsers (see [`requirements_treesitter.txt`](./Utils/requirements_treesitter.txt))
- Git CLI

**Hardware (recommended):**
- Content retrieval (Stage 4): high-core-count machine with fast network
- MinHash deduplication (Stage 5): single machine with large RAM (scales with dataset size and number of permutations)
- LLM scoring (Stage 7): GPU instance for batch API or vLLM inference
- Judge inference (Stage 8): multi-GPU node (tensor parallelism across available GPUs)

---

## 3. Stage 1 — Commit Mining (BigQuery)

**Script:** [`Commit_Mining_SQL/consolidated_query.sql`](./Commit_Mining_SQL/consolidated_query.sql)

Run this query in the Google BigQuery console or via the `bq` CLI. It extracts single-file commits from the GitHub public dataset. The query is a modified version of the BigQuery pipeline from [OctoPack](https://arxiv.org/abs/2308.07124) (Muennighoff et al., 2024), whose [CommitPack](https://huggingface.co/datasets/bigcode/commitpack) dataset pioneered large-scale commit mining from the same BigQuery source. The query restricts to **permissively licensed** repositories and the BigQuery snapshot contains commits up to **early 2022** — predating widespread LLM code generation — guaranteeing that all mined code is human-authored.

### 3.1 What the Query Does

The query joins four BigQuery tables:

| Table | Role |
|---|---|
| `github_repos.languages` | Maps repos to their declared programming languages |
| `github_repos.licenses` | Filters for permissive / open-source licenses |
| `github_repos.commits` | Source of commit metadata (hash, subject, message, timestamp, diffs) |
| `commits.difference` (unnested) | Per-file change records within each commit |

### 3.2 Filters Applied

| Filter | Purpose |
|---|---|
| **License allowlist** | Only permissive licenses: MIT, Apache-2.0, BSD-2/3-Clause, ISC, CC0-1.0, EPL-1.0, MPL-2.0, Unlicense, AGPL-3.0, LGPL-2.1, Artistic-2.0 |
| **Language allowlist** | Python, Java, JavaScript, C, C#, C++, TypeScript, Go, Ruby |
| **Message length** | `10 < LENGTH(message) < 15000` — removes empty/trivial and excessively long messages |
| **Message blocklist** | Excludes ~50 low-signal commit messages (`"initial commit"`, `"update readme.md"`, `"wip"`, `"yolo"`, etc.) |
| **Message pattern exclusion** | `NOT LIKE '%pi push%'`, `NOT LIKE '%push pi%'`, `NOT LIKE 'merge%'` |
| **Same-path constraint** | `old_path = new_path` — the file was modified in place, not renamed or moved |
| **Non-null paths** | Both `old_path` and `new_path` must exist |
| **Single-file constraint** | `HAVING COUNT(DISTINCT d.old_path) = 1` — the commit touches exactly one file |

### 3.3 Output Schema

| Column | Type | Description |
|---|---|---|
| `commit` | STRING | Git commit SHA |
| `subject` | STRING | First line of the commit message |
| `message` | STRING | Full commit message body |
| `repos` | STRING | Comma-separated list of repo names containing this commit |
| `license` | STRING | SPDX license identifier |
| `old_file` | STRING | Path of the changed file (pre-commit) |
| `new_file` | STRING | Path of the changed file (post-commit) |
| `unix_time` | INTEGER | Committer timestamp |

### 3.4 Running the Query

```bash
# Via bq CLI (exports to a BigQuery table)
bq query --use_legacy_sql=false --destination_table=your_project:your_dataset.single_file_commits \
  < Commit_Mining_SQL/consolidated_query.sql

# Export to GCS as newline-delimited JSON
bq extract --destination_format=NEWLINE_DELIMITED_JSON \
  your_project:your_dataset.single_file_commits \
  gs://your-bucket/single_file_commits/*.jsonl
```

---

## 4. Stage 2 — Repository Reputation Filtering

**Data:** [`Repos/`](./Repos/)

After exporting commits from BigQuery, filter them to keep only commits originating from repositories in the curated allowlists. This removes commits from low-quality, toy, or generated repositories and focuses the dataset on established, actively-maintained codebases.

### 4.1 Allowlist Contents

The [`Repos/`](./Repos/) directory contains pre-compiled lists of high-reputation repositories across 33 programming languages (~348k unique repos total). Each language has one JSON file (e.g. `python.json`) containing an array of `"owner/repo"` strings.

**Repo counts by language (top 10):**

| Language | Repos | Language | Repos |
|---|---|---|---|
| JavaScript | 52,041 | C# | 17,575 |
| Python | 41,494 | C | 17,395 |
| Java | 32,078 | Ruby | 16,581 |
| TypeScript | 31,093 | Rust | 11,037 |
| PHP | 24,507 | Swift | 7,616 |
| C++ | 23,687 | | |

The full list spans 33 languages including Go (22,688), Kotlin (5,697), Scala (3,217), Haskell (2,295), Erlang (1,152), and others down to Lisp (755).

### 4.2 How to Filter

The BigQuery output's `repos` column is a comma-separated string of `owner/repo` names. A commit passes the filter if **any** of its repos appears in the allowlist.

```python
import json
from pathlib import Path

def load_repo_allowlist(repos_dir: str = "Repos") -> set:
    """Load the union of all per-language repo lists into a single set."""
    allowlist = set()
    for path in Path(repos_dir).glob("*.json"):
        allowlist.update(json.loads(path.read_text()))
    return allowlist

allowlist = load_repo_allowlist("Repos")

def commit_in_allowlist(example: dict) -> bool:
    """Return True if any repo for this commit is in the allowlist."""
    commit_repos = {r.strip() for r in example["repos"].split(",")}
    return bool(commit_repos & allowlist)

# Apply as a HuggingFace filter
dataset = dataset.filter(commit_in_allowlist, num_proc=32)
```

### 4.3 Extending the Allowlists

To add new repos or languages, see [Section 12 — Building & Extending the Repo Allowlists](#12-auxiliary-building--extending-the-repo-allowlists).

---

## 5. Stage 3 — Extension Filtering

**Config:** [`Utils/programming_languages_ext_map.json`](./Utils/programming_languages_ext_map.json)

After repository filtering, further subset commits so the changed file's extension matches a target programming language. The extension map covers 27 languages:

```
Assembly (.asm), C (.c, .h), C# (.cs), C++ (.cpp, .c++, .cc, .cxx, .h++),
Common Lisp (.lisp), D (.di), Dart (.dart), Erlang (.erl, .hrl), Go (.go),
Groovy (.groovy), Haskell (.hs), Java (.java), JavaScript (.js),
Julia (.jl), Kotlin (.kt), PHP (.php), Perl (.pl), PowerShell (.ps1),
Python (.py), R (.r), Ruby (.rb), Rust (.rs), Scala (.scala),
Shell (.sh, .bash), Swift (.swift), TypeScript (.ts)
```

For each row, check that the extension of `new_file` appears in the extension list for one of the target languages. Rows that don't match are discarded.

---

## 6. Stage 4 — Content Retrieval

**Script:** [`Utils/retrieve_commit_contents.py`](./Utils/retrieve_commit_contents.py)

This script takes the filtered commit metadata (loaded as a HuggingFace dataset) and fetches the actual file contents — both the pre-commit (old) and post-commit (new) versions — directly from GitHub via shallow git fetches.

### 6.1 How It Works

For each commit:
1. Creates a temporary directory
2. Runs `git init` + `git remote add origin git@github.com:<repo>.git`
3. Runs `git fetch --depth 2 origin <commit_sha>` (shallow fetch with parent)
4. Checks out `FETCH_HEAD -- <new_file>` for the post-commit contents
5. Checks out `FETCH_HEAD^ -- <old_file>` for the pre-commit contents
6. Cleans up the temporary directory

If a commit appears in multiple repos (the `repos` field is comma-separated), the script tries each repo in order, stopping at the first successful fetch. It gives up after 15 failed repos for a single commit.

### 6.2 Arguments

| Argument | Default | Description |
|---|---|---|
| `--start_index` | 0 | Start row in the dataset |
| `--end_index` | 1000000 | End row in the dataset (exclusive) |
| `--workers` | 64 | Number of HuggingFace `map` processes |
| `--base_dir` | `/single-file-commits` | Working directory for temp repos and output |

### 6.3 Parallelism Model

The script uses a two-level parallelism model:
- **Outer level:** HuggingFace `dataset.map(num_proc=workers)` — spawns `workers` independent processes
- **Inner level:** Each process runs a `ThreadPoolExecutor` with 12 threads (`NUM_THREADS`), processing batches of 12 commits concurrently

This means the system can have up to `workers × 12` concurrent git operations. On a 64-worker setup, that's 768 simultaneous fetches.

### 6.4 Running

```bash
python retrieve_commit_contents.py \
  --start_index 0 \
  --end_index 500000 \
  --workers 64 \
  --base_dir /mnt/scratch/single-file-commits
```

**Output:** A JSONL file at `<base_dir>/diffs_<start>_to_<end>.jsonl` containing all original fields plus `new_contents`, `old_contents`, `returncode`, and `stderr`. Rows where either content is empty are filtered out.

### 6.5 Failure Modes

- **Auth-required repos:** `git fetch` returns non-zero → row gets empty contents and is filtered out
- **File-only commits (no parent):** `FETCH_HEAD^` checkout fails → `old_contents` is empty, `new_contents` is preserved
- **Timeouts:** Each shell command has a 120-second timeout; if exceeded, the commit is skipped via exception handling
- **Rate limiting:** GitHub may throttle SSH fetches. Running from multiple IPs or using a GitHub token can help.

### 6.6 Contrast with `scrape_git_repos.py`

Both scripts fetch data from GitHub but serve different purposes:

| | `retrieve_commit_contents.py` | `scrape_git_repos.py` |
|---|---|---|
| **Input** | Commit metadata (SHA, repos, file paths) | Repo metadata (repo name, optional revision) |
| **Method** | `git init` → `git fetch --depth 2` → checkout two file snapshots | `git clone --depth 1` → optionally `git reset` → `yek` serialise |
| **Output** | `old_contents` + `new_contents` (text of one changed file, before/after) | `serialised_repo` (entire repo tree as JSON) |
| **Pipeline role** | Stage 4 — retrieves the pre/post file contents for each mined commit | [Section 12](#12-auxiliary-building--extending-the-repo-allowlists) — builds/extends the `Repos/` allowlists |

---

## 7. Stage 5 — MinHash Deduplication

**Script:** [`Utils/minHash_dedupe_local.py`](./Utils/minHash_dedupe_local.py)

Removes near-duplicate content from the retrieved dataset using MinHash Locality-Sensitive Hashing. Adapted from the [BigCode deduplication pipeline](https://huggingface.co/blog/dedup).

### 7.1 Algorithm

1. **Shingling:** Tokenise each document by splitting on non-alphanumeric characters, then extract `ngram_size`-grams (default 5)
2. **MinHash:** For each document, compute `num_perm` hash permutations (default 256) to produce a signature vector
3. **LSH Banding:** Divide each signature into `B` bands of `R` rows. Documents sharing an identical band in any hash table are candidate duplicates
4. **Union-Find Clustering:** Candidate pairs are merged into clusters using a union-find structure
5. **Deduplication:** Within each cluster, only the representative (lowest-index) document is kept

The optimal `(B, R)` parameters are computed automatically to minimise a weighted sum of false-positive and false-negative probabilities at the given `threshold`.

### 7.2 Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `codeparrot/codeparrot-clean-valid` | Path to a HuggingFace dataset on disk (`load_from_disk`) |
| `--column` | `content` | Column name to deduplicate on |
| `--ngram-size` | 5 | N-gram size for shingling |
| `--num-perm` | 256 | Number of MinHash permutations |
| `--threshold` | 0.7 | Jaccard similarity threshold for deduplication |
| `--min-ngram-size` | 5 | Documents with fewer tokens than this are removed |
| `--output` | `None` | Output directory (creates `<output>/deduplicated/`) |
| `--map-parallelism` | 8 | Processes for the fingerprinting and cluster-assignment map stages |
| `--filter-parallelism` | 8 | Processes for the final filtering stage |
| `--num-samples` | `None` | Optionally subset to this many samples before deduplication |

### 7.3 Running

Run deduplication separately for `old_contents` and `new_contents` columns (or whichever content column you're targeting):

```bash
python minHash_dedupe_local.py \
  --dataset /mnt/data/retrieved_commits \
  --column new_contents \
  --threshold 0.7 \
  --num-perm 256 \
  --output /mnt/data/deduped_new \
  --map-parallelism 16 \
  --filter-parallelism 16
```

### 7.4 Memory Considerations

The clustering step loads all MinHash signatures and hash tables into memory. RAM usage scales with dataset size and number of permutations. The script disables garbage collection during the cluster-assignment phase (`gc.freeze()` / `gc.disable()`) to avoid performance degradation from the large union-find structure.

---

## 7.5. Pull Request Cross-Referencing (Commit Preferences Only)

Before term filtering, commit data destined for training preference pairs is cross-referenced with GHTorrent pull request data. Only commits that are part of subsequently successfully-merged, non-reverted pull requests are retained. All GitHub commit preference data used for training is sourced no later than **March 2019** and comes from a disjoint set of repositories vis-à-vis the commit data in Themis-CodeRewardBench.

---

## 8. Stage 6 — Term-Based Aspect Filtering & Commit Classification

**Directory:** [`Commit_Mining_Terms/`](./Commit_Mining_Terms/)

Aspect-specific filtering is a two-step process: (1) term-based retrieval of seed positives, followed by (2) training criteria-specialized [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) commit classifiers for high-confidence recall at scale. The term lists stored in this repository serve as the seed retrieval mechanism for step (1).

### 8.1 Step 1: Term-Based Seed Retrieval

Each `.list` file contains one lowercase search term per line, initially generated using frontier open-source LMs and then manually curated. A commit is assigned to an aspect if its `message` or `subject` (lowercased) contains any term from the corresponding list.

The term lists are **minimised**: any term that is a substring of another term in the same list has been removed, since the substring match already covers it. Exact duplicates have also been removed.

| File | Aspect | Term Count | Example Terms |
|---|---|---|---|
| [`functionality_term_list.list`](./Commit_Mining_Terms/functionality_term_list.list) | Functional Correctness | 135 | `"fix bug"`, `"edge case"`, `"undefined variable"`, `"handle exception"` |
| [`runtime_eff_term_list.list`](./Commit_Mining_Terms/runtime_eff_term_list.list) | Runtime Efficiency | 226 | `"optimize loop"`, `"cache results"`, `"reduce time complexity"`, `"speedup"` |
| [`memory_eff_term_list.list`](./Commit_Mining_Terms/memory_eff_term_list.list) | Memory Efficiency | 24 | `"memory leak"`, `"garbage collection"`, `"zero copy"`, `"lazy load"` |
| [`sec_term_list.list`](./Commit_Mining_Terms/sec_term_list.list) | Security Hardness | 271 | `"sql injection"`, `"xss"`, `"buffer overflow"`, `"cve"`, `"sanitiz"` |
| [`readability_term_list.list`](./Commit_Mining_Terms/readability_term_list.list) | Readability & Maintainability | 166 | `"refactor"`, `"dead code"`, `"code smell"`, `"rename"`, `"simplify"` |

**Matching strategy:** Terms are matched as substring containment on the lowercased commit message. Many terms use intentional prefix truncation (e.g. `"fix concurre"`, `"sanitiz"`, `"mem align"`) to capture morphological variants without requiring stemming. Because the lists are minimised, no term is redundant — every term in the list matches something that no other term in the same list already covers.

A single commit may match multiple aspects. Downstream stages handle this as multi-labelling.

```python
from pathlib import Path

def load_term_list(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]

sec_terms = load_term_list("Commit_Mining_Terms/sec_term_list.list")

def has_security_terms(message: str) -> bool:
    msg = message.lower()
    return any(term in msg for term in sec_terms)

# Apply as a HuggingFace filter
security_commits = dataset.filter(lambda ex: has_security_terms(ex["message"]))
```

### 8.2 Step 2: ModernBERT Commit Classifiers

The term-retrieved commits from Step 1 serve as labelled training data for criteria-specialized [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) commit classifiers — one per aspect. These classifiers are trained on the commit message text and used to recall high-confidence aspect-relevant commits from the full deduplicated corpus, including commits whose messages may not contain any of the seed terms but are nonetheless relevant to the criterion.

This two-step approach (term retrieval → classifier training → classifier inference) substantially increases recall over term matching alone, as the classifiers learn to generalise beyond the explicit seed terms to semantic patterns in commit messages.

**Note:** The classifier training code is not included in this repository. The term lists and the pipeline stages above and below this step are sufficient to reproduce the term-based seed retrieval; the classifier step is described here for completeness and to explain how the final aspect-labelled commit subsets in the training data were produced.

---

## 9. Stage 7 — LLM Scoring & Instruction Synthesis

**Templates:** [`Prompts/templates/`](./Prompts/templates/)

This stage uses Jinja2 prompt templates to construct LLM prompts from commit data. The rendered prompts are formatted as OpenAI batch API requests and submitted to an LLM endpoint. They serve two purposes: (1) validate that the code change actually improves the target aspect, and (2) generate a natural-language instruction that a developer could follow to produce the code.

All templates use Jinja2 `{% block system %}` / `{% block user %}` blocks to produce a system + user message pair (except `inverse_instruction.j2`, which is user-only). Literal Jinja2 braces in the LLM output scaffold are escaped as `{{'{{'}}` / `{{'}}'}}`.

### 9.1 `commit_review.j2` — Code Review & Rating

**Template:** [`templates/commit_review.j2`](./Prompts/templates/commit_review.j2)

Frames the LLM as a Principal Software Engineer reviewing a colleague's single-file code change along a specified quality axis. Produces four structured sections:

1. **[DESCRIPTION]** — What the pre-existing code does
2. **[SUMMARY]** — Factual summary of the technical changes
3. **[RATING]** — 1-5 score on whether the change improves the target aspect
4. **[FEEDBACK]** — Explanation of the score and improvement suggestions

| Variable | Type | Description |
|---|---|---|
| `aspect` | str | Quality axis under review (e.g. "Runtime Efficiency") |
| `language` | str | Programming language of the file |
| `old_contents` | str | File contents before the commit |
| `new_contents` | str | File contents after the commit |
| `message` | str | Commit message / change description |
| `diff` | str | Unified diff of the change |

**Rating scale:**

| Score | Meaning |
|---|---|
| 1 | Clear regression — degrades the target aspect |
| 2 | No discernible effect on the target aspect |
| 3 | Slight improvement; some unrelated edits |
| 4 | Significant improvement; mostly on-target |
| 5 | Must-have improvement; specific and well-implemented |

### 9.2 `commit_review_instruct.j2` — Code Review + Instruction Generation

**Template:** [`templates/commit_review_instruct.j2`](./Prompts/templates/commit_review_instruct.j2)

Two-phase prompt that extends the commit review with instruction synthesis. Phase 1 produces the same [DESCRIPTION], [SUMMARY], and [RATING] sections as `commit_review.j2` (without [FEEDBACK]). Phase 2 generates a contest-style problem instruction:

- The `problem_style` variable selects one of: `"graduate course assignment"`, `"quora question"`, `"stackoverflow question"`, `"google search query"`, `"programming contest problem"`
- The instruction must lead a developer to converge on either the old or new code equally
- Enclosed in `[INSTRUCTION]` / `[/INSTRUCTION]` tags

| Variable | Type | Description |
|---|---|---|
| `aspect` | str | Quality axis under review |
| `language` | str | Programming language |
| `old_contents` | str | File contents before the commit |
| `new_contents` | str | File contents after the commit |
| `message` | str | Commit message / change description |
| `diff` | str | Unified diff |
| `problem_style` | str | Style of the generated instruction (see above) |

### 9.3 `inverse_instruction.j2` — Instruction from Old/New Contents

**Template:** [`templates/inverse_instruction.j2`](./Prompts/templates/inverse_instruction.j2)

A lighter-weight template that takes (old_contents, new_contents) pairs and generates:

1. **[EXPLAIN]** — Common explanation of what both snippets accomplish
2. **[INSTRUCTION]** — A problem-style instruction that would lead to the code

This template does not perform quality scoring — it's purely for instruction synthesis. No system prompt is used (user-only message).

| Variable | Type | Description |
|---|---|---|
| `language` | str | Programming language |
| `old_contents` | str | File contents before the commit |
| `new_contents` | str | File contents after the commit |
| `problem_style` | str | Style of the generated instruction |

### 9.4 `problem_recovery.j2` — Problem Statement Recovery

**Template:** [`templates/problem_recovery.j2`](./Prompts/templates/problem_recovery.j2)

Given two reference solutions (chosen and rejected), the LLM reverse-engineers a problem statement that both solutions plausibly solve. Used to create instruction prompts for the preference dataset when the original problem is unavailable.

1. **[DESCRIPTION]** — Comprehensive description of what both code snippets accomplish
2. **[PROBLEM_STATEMENT]** — A clear, concise problem statement consistent with both solutions

| Variable | Type | Description |
|---|---|---|
| `language` | str | Programming language |
| `chosen` | str | Preferred/better solution code |
| `rejected` | str | Dispreferred/worse solution code |
| `problem_style` | str | Style of the generated problem statement |

### 9.5 `bug_introduction.j2` — Buggy Code Generation

**Template:** [`templates/bug_introduction.j2`](./Prompts/templates/bug_introduction.j2)

Given a validated (instruction, reference solution) pair, prompts an LLM to generate a functionally buggy variant of the reference solution. Used to create (buggy, correct) preference pairs for functional correctness training.

1. **[BUGGY_CODE]** — Modified reference solution with inserted bugs
2. **[BUG_EXPLANATION]** — Outline and explanation of the introduced bugs

Constraints enforced in the prompt:
- Only functional, logical, and algorithmic bugs (no security vulnerabilities, memory leaks, or non-functional bugs)
- Must maintain surface-level code structure
- Must not allude to the bugs in variable names, comments, or documentation

| Variable | Type | Description |
|---|---|---|
| `language` | str | Programming language |
| `instruction` | str | Problem statement / task description |
| `solution` | str | Correct reference solution code |

### 9.6 Rendering Templates & Batch Request Format

Templates are rendered with a Jinja2 environment and wrapped into OpenAI-compatible batch request objects:

```python
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader("Prompts/templates"))
template = env.get_template("commit_review.j2")

# Render the system and user blocks
rendered = template.render(
    aspect="Runtime Efficiency",
    language="Python",
    old_contents=example["old_contents"],
    new_contents=example["new_contents"],
    message=example["message"],
    diff=example["diff"],
)
```

The rendered system and user messages are then formatted as batch requests:

```json
{
  "custom_id": "<unique-id>",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "model": "<endpoint_name>",
    "messages": [
      {"role": "system", "content": "<rendered system block>"},
      {"role": "user", "content": "<rendered user block>"}
    ],
    "max_tokens": "<max_tokens>",
    "temperature": "<temperature>",
    "top_p": "<top_p>"
  }
}
```

These are stored as a new column in the dataset and can be exported as JSONL for submission to the OpenAI batch API.

---

## 10. Stage 8 — LLM-as-a-Judge Preference Labelling

**Template:** [`Prompts/templates/preference_judge.j2`](./Prompts/templates/preference_judge.j2)  
**Driver script:** [`Prompts/gen_llm.py`](./Prompts/gen_llm.py)

This stage runs a local vLLM instance to generate comparative preference judgements over the dataset. The prompt is defined in the `preference_judge.j2` Jinja2 template; `gen_llm.py` renders it, randomises A/B ordering, generates multiple independent judgements per sample, and aggregates by majority vote.

### 10.1 `preference_judge.j2` — Template Variables

The template produces a system + user message pair. The caller is responsible for randomising which of (chosen, rejected) maps to Completion A vs B and recording the mapping in a `chosen_index` field.

| Variable | Type | Description |
|---|---|---|
| `aspect` | str | Human-readable aspect name (e.g. "Readability and Maintainability") |
| `language` | str | Programming language, or `"NL"` for natural language |
| `criteria` | str | Numbered list of comparison criteria for the aspect (from `aspect_instructions` in `gen_llm.py`) |
| `prompt` | str | The original user prompt / task |
| `completion_a` | str | First completion (randomised order) |
| `completion_b` | str | Second completion (randomised order) |

The template sets `lang_tag = language.lower()` (or `"text"` when `language == "NL"`) and wraps completions in fenced code blocks using that tag.

### 10.2 Judging Protocol

1. **Randomised presentation:** For each sample, the chosen and rejected completions are randomly assigned to positions A and B (the `chosen_index` field records the mapping)
2. **Aspect-specific criteria:** The system prompt instructs the judge to evaluate on a specific aspect (Readability, Runtime Efficiency, Security, Functional Correctness, Memory Efficiency, Helpfulness, or Harmlessness) using 5-6 concrete comparison criteria
3. **Structured output:** The judge must produce `[EVALUATION]...[/EVALUATION]` (detailed rationale) followed by `[JUDGEMENT]A/B/TIE[/JUDGEMENT]`
4. **Multi-sample voting:** Each sample receives `num_responses` (default 8) independent judgements at temperature 0.8

### 10.3 Arguments

| Argument | Default | Description |
|---|---|---|
| `--input_dataset` | Required | HuggingFace dataset path or name |
| `--split` | `train` | Dataset split |
| `--output_prefix` | `judge_output` | Output directory prefix |
| `--model` | `Qwen/Qwen3-235B-A22B-Instruct-2507` | vLLM model for judge inference |
| `--start_index` / `--end_index` | None | Optional dataset sharding |
| `--num_responses` | 8 | Number of judge responses per sample |
| `--num_proc` | 24 | CPU processes for dataset mapping |
| `--tensor_parallel_size` | `torch.cuda.device_count()` | vLLM tensor parallelism |
| `--seed` | 77 | Random seed for sampling |

### 10.4 Running

```bash
python gen_llm.py \
  --input_dataset "your-org/commit-preference-scored" \
  --split train \
  --output_prefix "Judge_Outputs/Commit-Preference" \
  --model "Qwen/Qwen3-235B-A22B-Instruct-2507" \
  --start_index 0 \
  --end_index 8000 \
  --num_responses 8 \
  --num_proc 24 \
  --tensor_parallel_size 8 \
  --seed 77
```

### 10.5 Output Columns

After processing, each sample gains:

| Column | Description |
|---|---|
| `full_responses` | List of `num_responses` raw judge outputs |
| `rationales` | Extracted `[EVALUATION]` blocks |
| `num_matches` | Count of judgements agreeing with ground truth |
| `num_contradictions` | Count of judgements contradicting ground truth |
| `num_ties` | Count of TIE judgements |
| `num_errors` | Count of unparseable responses |

### 10.6 Intermediate Checkpoints

The script saves intermediate state at three points:
1. **Conversation-mapped dataset** → `<prefix>/intermediate_dataset[_shard_X_Y]`
2. **Raw vLLM outputs** (pickle) → `<prefix>/intermediate_output[_shard_X_Y].pkl`
3. **Final scored dataset** → `<prefix>/output_dataset[_shard_X_Y]`

This allows resuming from vLLM outputs if post-processing fails.

---

## 11. Stage 9 — System Prompt Assignment & Training Data Assembly

**Script:** [`Prompts/system_prompt_mapper.py`](./Prompts/system_prompt_mapper.py)

The final stage constructs the preference training dataset by assigning stochastic system prompts and computing tokenisation statistics.

### 11.1 System Prompt Strategy

Each training example receives one of three system prompt treatments, selected by a random draw:

| Probability Range | Treatment | Purpose |
|---|---|---|
| `[0, 0.15)` | **No system prompt** (`system = None`) | Teach the model to work without guidance |
| `[0.15, 0.75)` | **Aspect-specific prompt** (Helpfulness + Harmlessness + the sample's primary aspect) | Focused evaluation on the relevant axis |
| `[0.75, 1.0)` | **Full criteria prompt** (Helpfulness + Harmlessness + all 5 code aspects in random order) | Holistic evaluation across all axes |

The base prompt always includes Helpfulness and Harmlessness. For code aspects (Readability, Runtime Efficiency, Memory Efficiency, Security, Functional Correctness), the aspect-specific criterion is appended as criterion #3. The full prompt shuffles all five code criteria to avoid positional bias.

### 11.2 Output Columns

| Column | Description |
|---|---|
| `system` | The assigned system prompt (or `None`) |
| `Chosen_Conversation` | Full chat-template conversation with the chosen response |
| `Rejected_Conversation` | Full chat-template conversation with the rejected response |
| `Chosen_Conversation_Reduced` | Conversation without system prompt |
| `Rejected_Conversation_Reduced` | Conversation without system prompt |
| `*_Length` | Token counts after applying the target model's chat template |

### 11.3 Usage

```python
from transformers import AutoTokenizer
from Prompts.system_prompt_mapper import system_prompt_map

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")

dataset = dataset.map(
    system_prompt_map,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=24
)
```

---

## 11.4. Training Data Composition

### Themis-GeneralPreference (PT Stage — 110k+ samples)

A mixture of general-domain and code retrieval preferences used during Preference Model Pre-Training (PT) to instill common human-inspired notions of preference evaluation such as relevance, helpfulness, and harmlessness.

| Sub-Dataset | Samples | Description |
|---|---|---|
| [CodeR-Pile](https://huggingface.co/datasets/nebula2025/CodeR-Pile) | 41,924 | Code retrieval preferences from code augmentation, exemplar, integration, refinement, simplification, pseudo-code, tutorial, and web query subsets. Chosen from positive document; rejected via Zipfian sampling over hard negatives. |
| [Skywork-Preference](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2) | 30,146 | Filtered WildGuard, OffsetBias, Magpie, and HelpSteer2 subsets |
| [Tulu-IF](https://huggingface.co/datasets/allenai/tulu-3-pref-personas-instruction-following) | 13,705 | Instruction-following vs. instruction-violating response pairs |
| [H4-StackExchange](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences) | 12,139 | Highly-voted accepted answers vs. well-formed low-scoring answers |
| [Arena-HumanPreference](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) | 7,068 | Human voting validated preferences from LMArena head-to-head contests |
| [Prometheus-Preference](https://huggingface.co/datasets/prometheus-eval/Preference-Collection) | 2,864 | Helpfulness and harmlessness instruction clusters |
| [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) | 1,452 | High-margin preference pairs from general and STEM subsets |
| [Argilla-DPO](https://huggingface.co/datasets/argilla/distilabel-math-preference-dpo) | 1,042 | Math answer stylistic preferences (both responses converge on the same answer) |
| [Truthy-DPO](https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1) | 377 | Synthetically labeled truthfulness preferences |

### Themis-CodePreference (PM Stage — 350k+ samples)

Code-domain preferences across five quality dimensions and eight programming languages, used during Preference Modeling (PM).

| Sub-Dataset | Samples | Criteria | Languages |
|---|---|---|---|
| Commit-Preference | 126,586 | FC, RE, ME, SH, R&M | C, C#, C++, Go, Java, JS, Python, Ruby |
| [CodeR-Pile](https://huggingface.co/datasets/nebula2025/CodeR-Pile) | 77,153 | FC | C, C#, C++, Go, Java, JS, Python, Ruby |
| Bugged-Instruct | 54,969 | FC | C, C#, C++, Go, Java, JS, Python, Ruby |
| [ProSec](https://huggingface.co/datasets/prosecalign/prosec-mixed-clm7b-inst) | 37,690 | SH | C, C++, Java, JS, Python |
| [Venus](https://huggingface.co/datasets/Elfsong/Venus) | 21,784 | RE, ME | Python |
| [CodeNet](https://huggingface.co/datasets/iNeil77/CodeNet) | 16,819 | RE, ME | C, C#, C++, Go, Java, JS, Python, Ruby |
| [RunBugRun](https://huggingface.co/datasets/ASSERT-KTH/RunBugRun-Final) | 6,931 | FC | Python |
| [ECCO](https://huggingface.co/datasets/CodeEff/ECCO) | 6,188 | RE | Python |
| [CodeScaleR](https://huggingface.co/datasets/LARK-Lab/CodeScalerPair-51K) | 2,896 | FC | Python |
| [Pie4Perf](https://huggingface.co/datasets/iNeil77/pie4perf-Train) | 1,640 | RE | C++ |
| [Cybernative-DPO](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO) | 1,354 | SH | C#, C++, Go, Java, JS, Python, Ruby |

**Criteria key:** FC = Functional Correctness, RE = Runtime Efficiency, ME = Memory Efficiency, SH = Security Hardness, R&M = Readability & Maintainability.

### Data Filtering Procedure

Both Themis-GeneralPreference and Themis-CodePreference undergo the following filtering steps:

1. **Length filtering:** Samples exceeding 2,560 tokens (PT) or 4,096 tokens (PM) are removed. Code responses with syntax trees shallower than 3 levels deep are discarded.

2. **Language filtering:** Non-English prompts are discarded using the GlotLID language classifier. Prompts with perplexity >1,200 (measured by a KenLM model trained on OSCAR EN) are removed.

3. **Near-deduplication:** MinHash filtering with shingle size 20 and similarity threshold 0.75, applied separately per dataset.

4. **Benchmark decontamination:** Samples whose prompt registers a 13-gram overlap with any prompt in Themis-CodeRewardBench, RewardBench, or RM-Bench are removed.

---

## 12. Auxiliary: Building & Extending the Repo Allowlists

**Script:** [`Utils/scrape_git_repos.py`](./Utils/scrape_git_repos.py)  
**Data:** [`Repos/`](./Repos/)

The repo allowlists used in [Stage 2](#4-stage-2--repository-reputation-filtering) are pre-compiled in the `Repos/` directory. This section documents their structure and how to build or extend them using `scrape_git_repos.py`.

### 12.1 Allowlist File Format

Each language has a single JSON file — a flat array of `"owner/repo"` strings:

```json
// python.json
["pandas-dev/pandas", "scikit-learn/scikit-learn", ...]
```

32 per-language files covering ~348k repos total.

### 12.2 `scrape_git_repos.py` — Repo Cloning & Serialisation

This script clones repositories from GitHub and serialises their contents using the [`yek`](https://github.com/bodo-run/yek) CLI tool. While its primary output is the serialised repo JSON (useful for downstream context retrieval), the set of successfully-cloned repos also serves as the source for the `Repos/` allowlists.

**How it works:** For each row in the source HuggingFace dataset:
1. Clones the repo via `git clone --depth 1`
2. Optionally resets to a specific commit (`git reset --hard <revision_id>`) when `--reset_to_commit` is passed
3. Runs `yek --json --output-dir .` to serialise the repo into a single JSON file
4. Reads `yek-output-*.json` and stores it as the `serialised_repo` column
5. Repos that fail cloning, checkout, or serialisation are filtered out

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dataset` | Required | HuggingFace dataset name (e.g. `iNeil77/filtered-repos`) |
| `--output_prefix` | Required | Directory prefix for cloned repos and saved dataset (e.g. `Repos`) |
| `--reset_to_commit` | Disabled | After cloning, reset to the `revision_id` column value |
| `--start_index` | 0 | Start row in the dataset |
| `--end_index` | 103000 | End row (exclusive) |
| `--workers` | 20 | Number of HuggingFace `map` processes |

**Running:**

```bash
# Clone repos at a specific commit (e.g. for filtered-repos with revision_id)
python scrape_git_repos.py \
  --dataset "iNeil77/filtered-repos" \
  --output_prefix Repos \
  --reset_to_commit \
  --start_index 0 --end_index 103000 --workers 20

# Clone repos at HEAD only (e.g. for The Stack V2)
python scrape_git_repos.py \
  --dataset "iNeil77/the-stackv2-repo-python" \
  --output_prefix Repos5 \
  --start_index 0 --end_index 103000 --workers 20
```

### 12.3 Generating Allowlist Files from Scrape Output

After a scrape run completes, extract the repo names from the successfully-processed dataset to produce new `.list` and `.json` files:

```python
from datasets import load_from_disk
import json

ds = load_from_disk("Repos_0_to_103000")
repo_names = sorted(set(ds["repo_name"]))

with open("Repos/newlang.json", "w") as f:
    json.dump(repo_names, f, indent=2)
```

---

## 13. Auxiliary: Reward Model Evaluation

**Script:** [`Prompts/inference.py`](./Prompts/inference.py)

A standalone harness for evaluating a trained scalar reward model on the Code RewardBench (CRB) dataset. Not part of the dataset construction pipeline, but useful for validating the trained model.

### 13.1 What It Does

1. Loads a `AutoModelForSequenceClassification` reward model
2. Optionally prepends aspect-specific or full system prompts to conversations
3. Scores chosen and rejected responses in batches
4. Compiles accuracy breakdowns by aspect, language, and subset

### 13.2 Running

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py "your-org/ThemisRM-Inst-1.5B" \
  --output "ThemisRM-Inst-Aspect" \
  --use-system-prompts \
  --use-aspect-prompts \
  --batch-size 32 \
  --max-length 4096
```

### 13.3 System Prompt Modes

| Flag Combination | Behaviour |
|---|---|
| Neither flag | No system prompt — raw (user, assistant) conversations |
| `--use-system-prompts` only | Uses the **Full** system prompt (all 7 criteria) for every sample |
| `--use-system-prompts --use-aspect-prompts` | Uses the **aspect-specific** system prompt matching each sample's aspect |

### 13.4 Output

- `<output>/results.json` — accuracy statistics by aspect, language, and subset
- `<output>/scores.parquet` — per-sample chosen/rejected scores, correctness, and metadata
