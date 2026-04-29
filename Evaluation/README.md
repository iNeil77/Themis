# Evaluation Runbook

Evaluation harness for benchmarking reward models on the Code RewardBench (CRB) dataset. The suite supports a wide range of reward model architectures — scalar sequence classifiers, multi-objective MoE models, generative reward models, and more — all evaluated through a unified preference-accuracy protocol. Pre-computed results for 51 evaluated models are stored alongside the scripts.

**Related files:**

| Directory / File | Purpose |
|---|---|
| [`Evaluation_Scripts/`](./Evaluation_Scripts/) | Per-architecture evaluation scripts for CRB |
| [`Evaluation_Scripts/rerank_eval.py`](./Evaluation_Scripts/rerank_eval.py) | Code-completion reranking evaluation (separate task from CRB) |
| [`Evaluation_Runs/`](./Evaluation_Runs/) | Pre-computed results for 51 evaluated models (`results.json` + `scores.parquet` each) |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Benchmark Dataset](#3-benchmark-dataset)
4. [Evaluation Protocol](#4-evaluation-protocol)
5. [Script Categories](#5-script-categories)
6. [Standard Scalar RM — `coderewardbench-seqcls.py`](#6-standard-scalar-rm--coderewardbench-seqclspy)
7. [Custom-Architecture Scalar RMs](#7-custom-architecture-scalar-rms)
8. [Generative Reward Models](#8-generative-reward-models)
9. [Reranking Evaluation — `rerank_eval.py`](#9-reranking-evaluation--rerank_evalpy)
10. [Output Format](#10-output-format)
11. [Pre-Computed Results](#11-pre-computed-results)
12. [System Prompts](#12-system-prompts)

---

## 1. Overview

```
                        Code RewardBench (CRB)
                    ┌──────────────────────────┐
                    │ 8,866 preference pairs    │
                    │ (prompt, chosen, rejected) │
                    │ 5 aspects × 8 languages   │
                    │ 19 source subsets          │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                   ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │ Scalar RMs   │   │ MoE / Custom │   │ Generative   │
    │ (seqcls)     │   │ Architecture │   │ RMs (vLLM)   │
    │              │   │              │   │              │
    │ logits→score │   │ gating+heads │   │ parse text   │
    │              │   │ →score       │   │ →score       │
    └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
           │                  │                   │
           └──────────────────┼───────────────────┘
                              ▼
                    ┌──────────────────┐
                    │ chosen > rejected │
                    │ → correct / wrong │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ results.json     │  accuracy by aspect / language / subset
                    │ scores.parquet   │  per-sample scores and metadata
                    └──────────────────┘
```

Each script evaluates a reward model's ability to prefer the chosen response over the rejected response in each CRB sample. The model scores both responses independently; if `score(chosen) > score(rejected)`, the sample counts as correct.

---

## 2. Prerequisites

**Software:**
- Python 3.10+
- `torch`, `transformers`, `datasets`, `pandas`, `tqdm`
- `vllm` (for generative RM scripts only)
- `trl` (for GRM script only)
- `accelerate` (for LDL script only)
- `scipy`, `numpy` (for `rerank_eval.py`)

**Hardware:**
- GPU with sufficient VRAM for the model under evaluation
- Multi-GPU setups supported via `device_map="auto"` (scalar RMs) or `--tensor-parallel-size` (vLLM-based scripts)

**Data:**
- CRB dataset: `iNeil77/code-reward-bench` on HuggingFace (default config `"Full"`, split `"Full"`)
- Reranking dataset: `CodeShield/ccplus-rerank` on HuggingFace (for `rerank_eval.py`)

---

## 3. Benchmark Dataset

The Code RewardBench (CRB) dataset contains 8,866 preference pairs spanning 5 quality aspects and 8 programming languages, drawn from 19 source subsets.

### 3.1 Schema

| Column | Type | Description |
|---|---|---|
| `id` | str | Unique sample identifier |
| `prompt` | str | User query / coding task |
| `chosen` | str | Preferred response (ground truth) |
| `rejected` | str | Dispreferred response |
| `aspect` | str | Quality axis: `Functional_Correctness`, `Memory_Efficiency`, `Readability_Maintainability`, `Runtime_Efficiency`, `Security_Hardness` |
| `language` | str | Programming language: C, CSharp, Cpp, Go, Java, JavaScript, Python, Ruby |
| `subset` | str | Source dataset (e.g. `RUNBUGRUN`, `HUMANEVAL_PACK`, `COMMITPREFS_FUNCTIONAL`, etc.) |

### 3.2 Aspect Distribution

| Aspect | Count |
|---|---|
| Functional Correctness | 4,778 |
| Readability & Maintainability | 1,499 |
| Runtime Efficiency | 1,309 |
| Security Hardness | 991 |
| Memory Efficiency | 289 |

### 3.3 Source Subsets

19 subsets including: RUNBUGRUN (2,430), COMMITPREFS_READABILITY (1,371), COMMITPREFS_FUNCTIONAL (825), COMMITPREFS_SECURITY (769), DEBUG_EVAL (724), HUMANEVAL_PACK (628), PIE_PERF (460), ECCO (399), COMMITPREFS_MEMORY (252), COMMITPREFS_RUNTIME (238), EVALPERF (212), CODEPREFBENCH_SECURITY (173), MDEVAL (134), NOFUNEVAL_MAINTAIN (128), MBPP_PLUS_FIX_HARD (37), NOFUNEVAL_MEMORY (37), NOFUNEVAL_SECURITY (27), SECBENCH (14), VUL4J (8).

---

## 4. Evaluation Protocol

All scripts follow the same preference-accuracy protocol:

1. **Load** the CRB dataset from HuggingFace
2. **Format** each (prompt, response) pair as a chat conversation, optionally prepending a system prompt
3. **Score** both the chosen and rejected responses independently
4. **Compare**: a sample is **correct** if `score(chosen) > score(rejected)`
5. **Aggregate** accuracy overall and broken down by aspect, language, and subset
6. **Save** `results.json` (summary statistics) and `scores.parquet` (per-sample details)

The scoring mechanism differs by model architecture — scalar RMs extract a logit, MoE models combine gated multi-objective scores, and generative RMs parse scores from generated text.

---

## 5. Script Categories

The 20 evaluation scripts are organised into four categories based on model architecture:

| Category | Scripts | Scoring Method | Inference Engine |
|---|---|---|---|
| **Standard Scalar** | `seqcls` | `AutoModelForSequenceClassification` → `logits` | PyTorch |
| **Custom Scalar** | `armo`, `qrm`, `qrm-llama`, `athene`, `inform`, `acecode`, `ldl`, `grm`, `internlm`, `nemotron`, `starling`, `eurus`, `ultra`, `automodel` | Custom model class → architecture-specific score extraction | PyTorch |
| **Generative** | `cerm`, `nemotron-genrm`, `lmunit`, `r3` | vLLM text generation → parse score from output | vLLM |
| **Reranking** | `rerank_eval` | Scalar RM scores + ranking metrics (Hits@K, Spearman) | PyTorch |

### 5.1 Which Script for Which Model

| Script | Target Models | Key Distinction |
|---|---|---|
| `coderewardbench-seqcls.py` | Themis-RM-*, SkyworkV2-*, CodeScaler-*, FsfairX, Llama-3.1-*-RM-RB2, OffsetBias, URM, Tulu-3, WorldPM, Qwen2.5-Math-RM | Standard `AutoModelForSequenceClassification` — works for any model that exposes a scalar logit |
| `coderewardbench-armo.py` | ArmoRM-Llama3-8B-v0.1 | Llama + MoE gating network (19 objectives, learned temperature-scaled softmax gating) |
| `coderewardbench-qrm.py` | QRM-Gemma-2-27B | Gemma2 + quantile regression + MoE gating (5 objectives × 19 quantiles) |
| `coderewardbench-qrm-llama.py` | QRM-Llama3.1-8B-v2 | Same quantile + gating architecture but on Llama backbone |
| `coderewardbench-athene.py` | Athene-RM-8B, Athene-RM-70B | Llama + `v_head` linear layer, uses CLS token (ID 128003) for score extraction |
| `coderewardbench-inform.py` | INF-ORM-Llama3.1-70B | Llama + two-layer MLP score head (Linear → ReLU → Linear) |
| `coderewardbench-acecode.py` | AceCodeRM-7B, AceCodeRM-32B, AceMath-7B-RM, AceMath-72B-RM | ValueHead on Qwen2 backbone |
| `coderewardbench-ldl.py` | LDL-Reward-Gemma-2-27B-v0.1 | Gemma2 + multi-output NN for label distribution learning |
| `coderewardbench-grm.py` | GRM-llama3-8B-sftreg, GRM-llama3.2-3B-sftreg | `trl.PreTrainedModelWrapper` with ValueHead |
| `coderewardbench-internlm.py` | internlm2-* (1.8B, 7B, 20B variants) | Custom InternLM2 architecture with reward head |
| `coderewardbench-nemotron.py` | Llama-3.3-Nemotron-70B-Reward | Causal LM; extracts reward from helpfulness logprobs at the final token |
| `coderewardbench-starling.py` | Starling-RM-34B | Causal LM; custom reward extraction via forward pass on full conversation |
| `coderewardbench-eurus.py` | Eurus-RM-7b | `AutoModel` with custom reward token pooling |
| `coderewardbench-ultra.py` | UltraRM-13b | `AutoModel` with custom reward extraction |
| `coderewardbench-automodel.py` | mistral-7b-gsm8k-code-rm | Generic `AutoModel` fallback |
| `coderewardbench-cerm.py` | CE-RM-4B | Two-turn generative: criteria generation → evaluation; parses `\boxed{}` scores via vLLM |
| `coderewardbench-nemotron-genrm.py` | Qwen3-Nemotron-32B-GenRM-Principle | Generative with principle-based prompts; reward from logprobs of " Yes" / " No" tokens |
| `coderewardbench-lmunit.py` | LMUnit-qwen2.5-72b | Generative with aspect-specific unit tests; parses `{"score": N}` JSON output |
| `coderewardbench-r3.py` | R3-Qwen3-4B/8B/14B-14k | Rubric-based generative evaluation; parses numerical score from structured output |

---

## 6. Standard Scalar RM — `coderewardbench-seqcls.py`

The baseline evaluator for any model loadable via `AutoModelForSequenceClassification`. This is the most widely applicable script and covers the majority of evaluated models.

### 6.1 How It Works

1. Loads the model with `AutoModelForSequenceClassification` and `device_map="auto"`
2. For each sample, formats two conversations (chosen and rejected) using the tokenizer's chat template
3. Tokenizes and feeds each conversation through the model
4. Extracts the scalar reward from `outputs.logits`
5. Compares chosen vs rejected scores

### 6.2 Arguments

| Argument | Default | Description |
|---|---|---|
| `model_path` | Required (positional) | HuggingFace model ID or local path |
| `--dataset` | `iNeil77/code-reward-bench` | HuggingFace dataset name |
| `--config` | `None` | Dataset configuration (defaults to `"Full"`) |
| `--split` | `None` | Dataset split (defaults to `"Full"`) |
| `--output` | Required | Output directory for `results.json` and `scores.parquet` |
| `--max-length` | 4096 | Maximum sequence length for tokenization |
| `--batch-size` | 8 | Inference batch size |
| `--use-system-prompts` | `False` | Prepend a system prompt to each conversation |
| `--use-aspect-prompts` | `False` | Use aspect-specific system prompts (requires `--use-system-prompts`) |

### 6.3 Running

```bash
CUDA_VISIBLE_DEVICES=0 python coderewardbench-seqcls.py \
  "CodeShield/Themis-RM-8B" \
  --output "Themis-RM-8B" \
  --use-system-prompts \
  --use-aspect-prompts \
  --batch-size 32 \
  --max-length 4096
```

---

## 7. Custom-Architecture Scalar RMs

These scripts share the same CLI interface, `RewardModelEvaluator` class structure, and output format as `seqcls.py`, but each defines a custom model class inline to handle architectures that don't work with `AutoModelForSequenceClassification`.

### 7.1 MoE Gating Models

**Scripts:** `coderewardbench-armo.py`, `coderewardbench-qrm.py`, `coderewardbench-qrm-llama.py`

These models decompose the reward into multiple objectives and combine them via a learned gating network:

```
Hidden states → Regression layer → per-objective rewards
                                         ↓
Hidden states → Gating network   → per-objective weights (softmax)
                                         ↓
                              Weighted sum → final scalar score
```

**ArmoRM** (`armo`): Llama backbone, 19 objectives, gating on prompt-boundary token position (Llama `<|eot_id|><|start_header_id|>assistant` pattern). Score = `sum(gating * (rewards @ transform_matrix))`.

**QRM-Gemma** (`qrm`): Gemma2 backbone, 5 objectives × 19 quantiles, gating on prompt-boundary token position (Gemma2 `<end_of_turn>\n<start_of_turn>model` pattern). BatchNorm + dropout in gating MLP. Score = weighted mean of quantile-averaged rewards.

**QRM-Llama** (`qrm-llama`): Same quantile + gating concept on Llama backbone.

### 7.2 Custom Score Head Models

**Athene** (`athene`): Llama + single `v_head` linear layer. Extracts the score at the position of the CLS token (ID `128003`), not the last token. Appends `tokenizer.cls_token` to the formatted conversation.

**INFORM** (`inform`): Llama + two-layer MLP (Linear → ReLU → Linear) as the score head. Uses standard last-token pooling.

**AceCode** (`acecode`): Qwen2 causal LM with a `ValueHead` (two linear layers + ReLU + dropout). Score from last non-padding token.

**LDL** (`ldl`): Gemma2 + `MultiOutputNN` for label distribution learning. Score derived from the expected value of the learned distribution.

**GRM** (`grm`): Uses `trl.PreTrainedModelWrapper` with a `ValueHead` around a causal LM. Score from the value head's output at the last token.

### 7.3 Causal LM Score Extraction

**Nemotron** (`nemotron`): `AutoModelForCausalLM`. Extracts per-attribute logprobs (helpfulness, correctness, etc.) at a designated token position. The helpfulness logprob serves as the scalar reward.

**Starling** (`starling`): `AutoModelForCausalLM` with custom forward pass for reward extraction.

**Eurus** (`eurus`), **UltraRM** (`ultra`), **AutoModel** (`automodel`): `AutoModel` variants with model-specific pooling and linear projection for score extraction.

**InternLM** (`internlm`): Fully custom InternLM2 architecture (custom config, custom model class with reward head) loaded without `trust_remote_code` — the architecture is defined entirely within the script.

### 7.4 Running Custom Scripts

All custom scripts accept the same CLI arguments as `seqcls.py`:

```bash
# ArmoRM with MoE gating
CUDA_VISIBLE_DEVICES=0 python coderewardbench-armo.py \
  "RLHFlow/ArmoRM-Llama3-8B-v0.1" \
  --output "ArmoRM-Llama3-8B-v0.1" \
  --batch-size 16 --max-length 4096

# Athene with CLS-token scoring
CUDA_VISIBLE_DEVICES=0 python coderewardbench-athene.py \
  "Nexusflow/Athene-RM-70B" \
  --output "Athene-RM-70B" \
  --use-system-prompts --use-aspect-prompts \
  --batch-size 8 --max-length 4096
```

---

## 8. Generative Reward Models

These scripts use vLLM for batched text generation and parse numerical scores from the model's textual output. Each has a distinct prompting strategy and score extraction method.

### 8.1 CE-RM — `coderewardbench-cerm.py`

**Two-turn criteria-based evaluation.** For each (prompt, response) pair:
1. **Turn 1 (Criteria):** Generates a minimal set of evaluation criteria for the query
2. **Turn 2 (Evaluation):** Scores the response using those criteria; emits per-criterion scores in `\boxed{}` (0-5) and a final overall score in `\boxed{}` (0-10)

The last `\boxed{N}` value is extracted as the score. Parse failures default to 0.0.

| Argument | Default | Description |
|---|---|---|
| `model_path` | Required | HuggingFace model ID |
| `--batch-size` | 16 | Preference pairs per vLLM batch |
| `--tensor-parallel-size` | 1 | GPUs for tensor parallelism |
| `--max-new-tokens-criteria` | 4096 | Token budget for criteria generation |
| `--max-new-tokens-evaluation` | 8192 | Token budget for evaluation |
| `--max-model-len` | None | Cap on vLLM KV-cache context length |

```bash
CUDA_VISIBLE_DEVICES=0 python coderewardbench-cerm.py \
  "PKU-ONELab/CE-RM-4B" \
  --output "CE-RM-4B" \
  --batch-size 32 --tensor-parallel-size 1
```

### 8.2 Nemotron GenRM — `coderewardbench-nemotron-genrm.py`

**Principle-based Yes/No evaluation.** Formats conversations with a `principle` role message specifying the evaluation criteria, then derives the reward from logprobs of " Yes" (token 7414) vs " No" (token 2308) at the penultimate generation step. If a token falls outside the top-k logprobs, its logprob is clamped to -50.

Supports aspect-specific principle prompts via `--use-aspect-prompts` (e.g., "functional correctness" vs the full 7-criteria principle).

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python coderewardbench-nemotron-genrm.py \
  "nvidia/Qwen3-Nemotron-32B-GenRM-Principle" \
  --output "GenRM-Principle-vLLM" \
  --use-aspect-prompts \
  --max-model-len 32768 \
  --tensor-parallel-size 4
```

### 8.3 LMUnit — `coderewardbench-lmunit.py`

**Unit-test-style evaluation.** Formats each sample as `"Query: ... Response: ... Unit Test: ..."` where the unit test is an aspect-specific question (e.g., "Does the response correctly implement the requested functionality, handle edge cases properly, and produce the expected outputs?"). Parses the score from JSON output `{"score": N}`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python coderewardbench-lmunit.py \
  "ContextualAI/LMUnit-qwen2.5-72b" \
  --output "LMUnit-Aspect" \
  --use-aspect-prompts \
  --batch-size 256 --max-length 4096 \
  --tensor-parallel-size 4 --max-new-tokens 2048
```

### 8.4 R3 (Rubric Reward) — `coderewardbench-r3.py`

**Rubric-based evaluation.** Provides the model with aspect-specific rubric criteria and asks for a structured evaluation with a numerical score. Score extracted from the generated output via regex.

```bash
CUDA_VISIBLE_DEVICES=6,7 python coderewardbench-r3.py \
  "rubricreward/R3-Qwen3-14B-14k" \
  --output "R3-Qwen3-14B-14k" \
  --use-system-prompts --use-aspect-prompts \
  --batch-size 32 --max-model-len 10000 \
  --tensor-parallel-size 2
```

---

## 9. Reranking Evaluation — `rerank_eval.py`

A separate evaluation task that measures a scalar RM's ability to rank code completions by functional correctness.

### 9.1 Task

Given a coding question with multiple completions (some passing tests, some failing), the script:
1. Scores every completion with the reward model
2. Reports three metric families:
   - **Reward variance** per question (higher → more discriminative model)
   - **Hits@K** (K=5, 10): whether a passing completion appears in the top-K by reward score
   - **Rank correlation** (Spearman) between test-case pass-rate and reward scores

### 9.2 Arguments

| Argument | Default | Description |
|---|---|---|
| `model_path` | Required | Scalar RM (uses `AutoModelForSequenceClassification`) |
| `--dataset` | `CodeShield/ccplus-rerank` | HuggingFace reranking dataset |
| `--config` | Required | Language config (e.g. `"python"`) |
| `--output` | Required | Output directory |
| `--use-system-prompt` | `False` | Prepend Functional Correctness system prompt |
| `--batch-size` | 8 | Inference batch size |
| `--max-length` | 4096 | Maximum sequence length |

### 9.3 Running

```bash
CUDA_VISIBLE_DEVICES=0,1 python rerank_eval.py \
  "CodeShield/Themis-RM-32B" \
  --dataset "CodeShield/ccplus-rerank" \
  --config "python" \
  --output results_py \
  --use-system-prompt \
  --batch-size 8 --max-length 4096
```

---

## 10. Output Format

All CRB evaluation scripts produce two output files in the `--output` directory:

### 10.1 `results.json`

Aggregate accuracy statistics:

```json
{
  "overall": {
    "accuracy": 0.8978,
    "total_examples": 8866,
    "correct_predictions": 7960,
    "mean_score_difference": 2.8353,
    "median_score_difference": 1.9543,
    "std_score_difference": 3.0097
  },
  "by_aspect": {
    "Functional_Correctness": {"accuracy": 0.9175, "count": 4778},
    "Memory_Efficiency": {"accuracy": 0.9204, "count": 289},
    ...
  },
  "by_language": {
    "Python": {"accuracy": 0.8882, "count": 2566},
    ...
  },
  "by_subset": {
    "RUNBUGRUN": {"accuracy": 0.8951, "count": 2430},
    ...
  }
}
```

### 10.2 `scores.parquet`

Per-sample scores indexed by `id`:

| Column | Type | Description |
|---|---|---|
| `id` (index) | str | Sample identifier |
| `chosen_score` | float | Reward score for the chosen response |
| `rejected_score` | float | Reward score for the rejected response |
| `correct` | bool | Whether `chosen_score > rejected_score` |
| `score_diff` | float | `chosen_score - rejected_score` |
| `language` | str | Programming language |
| `aspect` | str | Quality aspect |
| `subset` | str | Source subset |

---

## 11. Pre-Computed Results

The `Evaluation_Runs/` directory contains results for 51 evaluated models. Each model has its own subdirectory with `results.json` and `scores.parquet`.

### 11.1 Evaluated Models

| Model | Size | Script Used | Architecture |
|---|---|---|---|
| **Themis-RM** (0.6B, 1.7B, 4B, 8B, 14B, 32B) | 0.6B–32B | `seqcls` | Standard `SeqCls` |
| **CodeScaler** (1.7B, 4B, 8B) | 1.7B–8B | `seqcls` | Standard `SeqCls` |
| **SkyworkV2-Qwen** (0.6B, 1.7B, 4B, 8B) | 0.6B–8B | `seqcls` | Standard `SeqCls` |
| Skywork-Reward-Gemma-2-27B-v0.2 | 27B | `seqcls` | Standard `SeqCls` |
| Llama-3.1-8B-Instruct-RM-RB2 | 8B | `seqcls` | Standard `SeqCls` |
| Llama-3.1-70B-Instruct-RM-RB2 | 70B | `seqcls` | Standard `SeqCls` |
| Llama-3.1-Tulu-3-70B-SFT-RM-RB2 | 70B | `seqcls` | Standard `SeqCls` |
| Llama-3-OffsetBias-RM-8B | 8B | `seqcls` | Standard `SeqCls` |
| FsfairX-LLaMA3-RM-v0.1 | 8B | `seqcls` | Standard `SeqCls` |
| URM-LLaMa-3.1-8B | 8B | `seqcls` | Standard `SeqCls` |
| WorldPM-72B-RLHFLow | 72B | `seqcls` | Standard `SeqCls` |
| Qwen2.5-Math-RM-72B | 72B | `seqcls` | Standard `SeqCls` |
| ArmoRM-Llama3-8B-v0.1 | 8B | `armo` | Llama + MoE gating (19 obj) |
| QRM-Gemma-2-27B | 27B | `qrm` | Gemma2 + quantile + MoE gating |
| QRM-Llama3.1-8B-v2 | 8B | `qrm-llama` | Llama + quantile + MoE gating |
| Athene-RM-8B | 8B | `athene` | Llama + v_head (CLS token) |
| Athene-RM-70B | 70B | `athene` | Llama + v_head (CLS token) |
| INF-ORM-Llama3.1-70B | 70B | `inform` | Llama + MLP score head |
| AceCodeRM-7B | 7B | `acecode` | Qwen2 + ValueHead |
| AceCodeRM-32B | 32B | `acecode` | Qwen2 + ValueHead |
| AceMath-7B-RM | 7B | `acecode` | Qwen2 + ValueHead |
| AceMath-72B-RM | 72B | `acecode` | Qwen2 + ValueHead |
| LDL-Reward-Gemma-2-27B-v0.1 | 27B | `ldl` | Gemma2 + MultiOutput NN |
| GRM-llama3-8B-sftreg | 8B | `grm` | trl ValueHead |
| GRM-llama3.2-3B-sftreg | 3B | `grm` | trl ValueHead |
| **internlm2** (1.8B, 7B, 7B-code-100k, 7B-math-100k, 20B) | 1.8B–20B | `internlm` | Custom InternLM2 |
| Llama-3.3-Nemotron-70B-Reward | 70B | `nemotron` | Causal LM logprob extraction |
| Starling-RM-34B | 34B | `starling` | Causal LM |
| Eurus-RM-7b | 7B | `eurus` | AutoModel + custom pooling |
| UltraRM-13b | 13B | `ultra` | AutoModel + custom pooling |
| mistral-7b-gsm8k-code-rm | 7B | `automodel` | AutoModel generic |
| CE-RM-4B | 4B | `cerm` | Generative (vLLM) |
| Qwen3-Nemotron-32B-GenRM-Principle | 32B | `nemotron-genrm` | Generative (vLLM) |
| LMUnit-qwen2.5-72b | 72B | `lmunit` | Generative (vLLM) |
| **R3-Qwen3** (4B, 8B, 14B) | 4B–14B | `r3` | Generative (vLLM) |

### 11.2 Loading Results

```python
import json
import pandas as pd

# Load aggregate results
with open("Evaluation_Runs/Themis-RM-8B/results.json") as f:
    results = json.load(f)
print(f"Overall accuracy: {results['overall']['accuracy']:.2%}")

# Load per-sample scores
scores = pd.read_parquet("Evaluation_Runs/Themis-RM-8B/scores.parquet")
print(scores.groupby("aspect")["correct"].mean())
```

### 11.3 Comparing Models

```python
import pandas as pd
from pathlib import Path
import json

rows = []
for model_dir in sorted(Path("Evaluation_Runs").iterdir()):
    results_path = model_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            r = json.load(f)
        rows.append({"model": model_dir.name, **r["overall"]})

leaderboard = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
print(leaderboard[["model", "accuracy", "total_examples"]].to_string(index=False))
```

---

## 12. System Prompts

All scripts share the same `SYSTEM_PROMPT_MAP` dictionary with 7 system prompt variants. The system prompt strategy is controlled by two flags:

| Flag Combination | Behaviour |
|---|---|
| Neither flag | No system prompt — raw `(user, assistant)` conversations |
| `--use-system-prompts` only | Prepends the **Full** system prompt (all 7 criteria) to every sample |
| `--use-system-prompts --use-aspect-prompts` | Prepends the **aspect-specific** system prompt matching each sample's `aspect` field |

### 12.1 Available Prompts

| Key | Criteria Included |
|---|---|
| `Full` | Helpfulness, Harmlessness, Memory Efficiency, Functional Correctness, Readability & Maintainability, Runtime Efficiency, Security Hardness |
| `Functional_Correctness` | Helpfulness, Harmlessness, Functional Correctness |
| `Memory_Efficiency` | Helpfulness, Harmlessness, Memory Efficiency |
| `Readability_Maintainability` | Helpfulness, Harmlessness, Readability & Maintainability |
| `Runtime_Efficiency` | Helpfulness, Harmlessness, Runtime Efficiency |
| `Security_Hardness` | Helpfulness, Harmlessness, Security Hardness |

Helpfulness and Harmlessness are always included as base criteria. The aspect-specific prompt adds only the relevant code quality criterion as criterion #3.

The generative RM scripts use analogous but format-specific variants: `PRINCIPLE_MAP` for Nemotron GenRM, `ASPECT_UNIT_TEST_MAP` for LMUnit, and `RUBRIC_MAP` for R3.
