# Reward Model Training Runbook

End-to-end guide for training a Bradley-Terry scalar reward model on multi-node GPU clusters using FSDP2, Liger fused kernels, and containerised environments.

**Related files:**

| File | Purpose |
|------|---------|
| [`train_reward_model.py`](./train_reward_model.py) | Training script — model, loss (fused linear CE), data pipeline, training loop |
| [`fsdp2_config.yaml`](./fsdp2_config.yaml) | Accelerate/FSDP2 distributed training configuration |
| [`launch_reward_training.sh`](./launch_reward_training.sh) | Per-node launch script (container runtime + accelerate) |
| [`Themis.Dockerfile`](./Themis.Dockerfile) | Container build definition (PyTorch + high-speed networking + Liger) |

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Container Setup](#2-container-setup)
3. [Cluster & Networking](#3-cluster--networking)
4. [Environment Variables](#4-environment-variables)
5. [Accelerate Config (FSDP2)](#5-accelerate-config-fsdp2)
6. [Training Script Arguments](#6-training-script-arguments)
7. [Launch (8 Nodes, 64 GPUs)](#7-launch-8-nodes-64-gpus)
8. [Monitoring & Debugging](#8-monitoring--debugging)
9. [Checkpointing & Recovery](#9-checkpointing--recovery)
10. [Adapting for a Different Model](#10-adapting-for-a-different-model)
11. [Inference After Training](#11-inference-after-training)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

**Hardware:**
- Multi-GPU nodes with 8x H100 80GB or 8x A100 40/80GB per node
- High-speed inter-node networking (e.g. InfiniBand, RoCE, or equivalent)
- Shared filesystem mounted at `/mnt/fsx/` (or equivalent network storage)
- NVMe instance storage available at `/opt/dlami/nvme/` (or equivalent local scratch)

**Software (on host):**
- NVIDIA driver with high-speed networking kernel module loaded
- Container runtime (e.g. enroot, Docker, Singularity)
- Job scheduler: Slurm or manual SSH orchestration

**Accounts & Tokens:**
- HuggingFace account with access to the base model (e.g. `Qwen/Qwen3-14B`)
- Weights & Biases account for experiment tracking

---

## 2. Container Setup

The training environment is packaged as a Docker image containing PyTorch, CUDA, high-speed networking drivers, NCCL, Liger Kernel, and all Python dependencies. See [`Themis.Dockerfile`](./Themis.Dockerfile) for the full build definition.

### 2.1 Building the Image (Optional)

If you need to rebuild (e.g. to update package versions):

```bash
docker build -f Themis.Dockerfile -t ineil77/themis:29042026-3 .
docker push ineil77/themis:29042026-3
```

The Dockerfile uses a single-stage build: high-speed networking libraries (libfabric, GDRCopy, OFI-NCCL plugin) are all compiled in-place to ensure proper library linking. Size is kept lean via debug symbol stripping, pip cache purging, and build-dependency removal.

### 2.2 Importing into enroot

On each compute node (or on a shared filesystem visible to all nodes):

```bash
# Pull from Docker Hub and convert to squashfs
enroot import docker://ineil77/themis:29042026-3
```

This produces `ineil77+themis+29042026-3.sqsh`. If using a shared filesystem, import once and let all nodes access the same `.sqsh` file.

### 2.3 Creating the Container

```bash
enroot create --name Themis ineil77+themis+29042026-3.sqsh
```

### 2.4 Verification

```bash
# Check the container exists
enroot list
# Expected output includes: Themis

# Quick sanity check — verify PyTorch sees GPUs
enroot start --root Themis python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
# Expected: GPUs: 8
```

---

## 3. Cluster & Networking

### 3.1 Node Topology

For an 8-node job:
- 64 GPUs total (8 per node)
- Intra-node: NVLink
- Inter-node: high-speed RDMA networking

### 3.2 Network Configuration

Ensure the network allows:
- All TCP/UDP traffic between nodes in the cluster (for NCCL)
- RDMA traffic between nodes (if using high-speed networking)

### 3.3 Shared Filesystem Layout

```
/mnt/fsx/
  Themis/v1.1/
    Standalone_Trainer/
      train_reward_model.py        # Training script
      fsdp2_config.yaml            # Accelerate config
      launch_reward_training.sh    # Launch script
    27B/
      Logs/                        # Accelerate log output (one dir per rank)
      Outputs/                     # Checkpoints and final model
```

---

## 4. Environment Variables

Export these on **every node** before launching. Your job scheduler (Slurm, etc.) should set the distributed coordination variables automatically.

```bash
# ---- Authentication ----
export HF_TOKEN="<your-huggingface-token>"
export WANDB_API_KEY="<your-wandb-api-key>"

# ---- Distributed coordination ----
# MASTER_ADDR: IP address of the rank-0 node (must be reachable from all other nodes)
export MASTER_ADDR="<ip-of-rank-0-node>"
# MASTER_PORT: any free port; all nodes must use the same value
export MASTER_PORT="29500"
# MACHINE_RANK: unique integer per node, 0-indexed (rank 0 is the master)
export MACHINE_RANK="<0-7>"
```

### Slurm Example

If using Slurm with `srun`, these are typically set automatically:

```bash
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export MACHINE_RANK=$SLURM_NODEID
```

---

## 5. Accelerate Config (FSDP2)

The distributed training strategy is configured in [`fsdp2_config.yaml`](./fsdp2_config.yaml). This file is passed to `accelerate launch --config_file` and controls how model parameters, gradients, and optimizer states are sharded across GPUs.

### 5.1 Full Configuration

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_version: 2
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_cpu_ram_efficient_loading: false
  fsdp_offload_params: false
  fsdp_reshard_after_forward: true
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: Qwen3DecoderLayer
main_training_function: main
mixed_precision: bf16
num_machines: 8
num_processes: 64
rdzv_backend: c10d
same_network: true
use_cpu: false
```

### 5.2 Key Fields

| Field | Description | When to Change |
|-------|-------------|----------------|
| `num_machines` | Number of physical nodes | Always — must match your job allocation |
| `num_processes` | Total GPUs = `num_machines` x GPUs/node | Always — e.g. 8 GPUs/node on H100/A100 systems |
| `fsdp_transformer_layer_cls_to_wrap` | The decoder layer class to shard | When switching model architectures (see [Section 10](#10-adapting-for-a-different-model)) |
| `fsdp_reshard_after_forward` | `true` = ZeRO-3 (lower memory); `false` = ZeRO-2 (lower communication) | Set `false` if you have memory headroom and want faster throughput |
| `fsdp_offload_params` | Offload sharded params to CPU RAM | Only for extremely large models that OOM even with full sharding |
| `fsdp_cpu_ram_efficient_loading` | Stream weights shard-by-shard during init | Set `true` if the full unsharded model doesn't fit in one GPU's memory at init time |
| `fsdp_state_dict_type` | How checkpoints are saved | `FULL_STATE_DICT` (rank-0 gathers all); `SHARDED_STATE_DICT` (per-rank, faster but requires same world size to resume) |
| `mixed_precision` | `bf16` for BFloat16 mixed precision | Use `fp16` only on pre-Ampere hardware; `no` for full fp32 (not recommended) |

### 5.3 How FSDP2 Sharding Works

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Layer N forward:                                                         │
│   1. All-gather: collect parameter shards from all ranks → full params   │
│   2. Compute: run forward pass for this layer                            │
│   3. Reshard: discard full params, keep only local shard                 │
│      (if fsdp_reshard_after_forward=true)                                │
│                                                                          │
│ Layer N backward:                                                        │
│   1. All-gather: collect params again for gradient computation           │
│   2. Compute: backward pass, produce gradients                           │
│   3. Reduce-scatter: each rank gets its shard of the gradient            │
│   4. Optimizer step: update local parameter shard                        │
└──────────────────────────────────────────────────────────────────────────┘
```

Each transformer layer is an independently-sharded FSDP unit, so peak memory is proportional to ~1 full layer + activations, not the entire model.

---

## 6. Training Script Arguments

Full reference for [`train_reward_model.py`](./train_reward_model.py). The script trains a reward model with a Bradley-Terry preference loss, LM regularisation, and magnitude penalty.

### 6.1 Required

| Argument | Description |
|----------|-------------|
| `--model_name_or_path` | HuggingFace model ID or local path to a pretrained causal LM |

### 6.2 Dataset

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_name` | `project-themis/Themis-GeneralPreference` | HuggingFace dataset with columns: `input`, `chosen`, `rejected`, `system` |
| `--dataset_split` | `train` | Dataset split to use (e.g. `train`, `test_rewardbench`) |
| `--filter_language` | None | Keep only examples with this language label (e.g. `Python`, `Java`, `NL`) |
| `--filter_aspect` | None | Keep only examples with this aspect label (e.g. `Helpfulness`, `Harmlessness`) |
| `--max_length` | 1024 | Maximum token count for prompt + response combined |
| `--truncate_response` | Enabled | When over `max_length`, truncate only the response (preserving the full prompt) |
| `--no_truncate_response` | | Drop over-length examples entirely instead of truncating |
| `--num_proc` | 8 | Number of CPU processes for dataset map/filter operations. Avoid using all cores on large machines to prevent memory pressure. |
| `--system_prompt` | None | Fallback system prompt for examples that don't have a per-example `system` field |

**Dataset processing:** All ranks process the dataset independently (no barriers or shared caching). HuggingFace datasets' built-in fingerprint cache means repeated runs on the same node hit cache automatically. Model weight downloads use `local_main_process_first()` so that one process per node downloads from HuggingFace Hub and the others load from the node-local cache.

### 6.3 Loss Function

The training loss combines three terms: `total = BT + lambda_lm * LM + lambda_mag * MAG`

The chosen sequence uses a **single forward pass** through the base transformer: hidden states feed both the reward head (last-token scalar) and the LM head (completion-token cross-entropy via `LigerFusedLinearCrossEntropyLoss`, which never materializes the full `(B, T, V)` logits tensor). The LM loss is computed on the same sequence as the reward — including system prompt if present. The rejected sequence only needs a reward, so it uses the base transformer without the LM head.

| Argument | Default | Description |
|----------|---------|-------------|
| `--lambda_lm` | 0.1 | Weight for LM regularisation. Keeps the backbone's generative ability intact. Higher values = stronger constraint but slower reward learning. |
| `--lambda_mag` | 0.01 | Weight for reward magnitude penalty `(|r_w| + |r_l|)^2`. Prevents unbounded reward growth. |

**Tuning guidance:**
- If reward accuracy converges but the model degrades at generation: increase `lambda_lm`
- If rewards grow very large (>10) or very negative: increase `lambda_mag`
- For most runs: `lambda_lm=0.1-0.5`, `lambda_mag=0.005-0.02`

### 6.4 Optimisation

| Argument | Default | Description |
|----------|---------|-------------|
| `--learning_rate` | 1e-5 | Peak learning rate (after warmup) |
| `--weight_decay` | 0.01 | AdamW L2 regularisation |
| `--per_device_train_batch_size` | 2 | Number of preference pairs per GPU per micro-step |
| `--gradient_accumulation_steps` | 8 | Micro-steps accumulated before one optimizer update |
| `--num_train_epochs` | 1 | Number of full passes over the dataset |
| `--gradient_checkpointing` | Enabled | Trade compute for memory by recomputing activations in backward |
| `--bf16` | Enabled | BFloat16 mixed-precision training |
| `--seed` | 42 | Random seed for reproducibility |

**Effective batch size:**
```
effective_batch = per_device_train_batch_size x num_processes x gradient_accumulation_steps

Example (this config): 4 x 64 x 4 = 1024 preference pairs per optimizer step
```

### 6.5 LR Schedule (Warmup-Stable-Decay)

The learning rate follows a three-phase WSD schedule:

```
LR
 ^
 |        ┌────────────────────────────┐
 |       /│         stable             │\
 |      / │                            │  \
 |     /  │                            │    \
 |    /   │                            │      \
 |   /    │                            │        \___min_lr
 └──/─────┼────────────────────────────┼────────────────> steps
    warmup          stable                decay
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--warmup_ratio` | 0.02 | Fraction of total steps for warmup ramp |
| `--stable_ratio` | 0.75 | Fraction of total steps at constant peak LR |
| `--decay_ratio` | 0.23 | Fraction of total steps for cooldown decay |
| `--warmup_type` | `linear` | Warmup curve shape: `linear`, `cosine`, `1-sqrt` |
| `--decay_type` | `cosine` | Decay curve shape: `linear`, `cosine`, `1-sqrt` |
| `--min_lr_ratio` | 0.0 | Floor LR as fraction of peak (0.0 = decay to zero) |

**Note:** `warmup_ratio + stable_ratio + decay_ratio` should sum to 1.0. The script computes actual step counts from these ratios.

### 6.6 Checkpointing & Logging

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `./reward_model_output` | Directory for checkpoints and final model |
| `--save_steps` | 500 | Save a checkpoint every N optimizer steps |
| `--save_epochs` | Disabled | Additionally save at the end of each epoch |
| `--logging_steps` | 10 | Log metrics to tracker every N steps |
| `--report_to` | `wandb` | Experiment tracker: `wandb`, `tensorboard`, `all`, `none` |
| `--wandb_project` | `reward-model-training` | Weights & Biases project name |
| `--wandb_run_name` | Auto-generated | Weights & Biases run name |

**Checkpoint format:** Each checkpoint is saved as a standard `AutoModelForSequenceClassification` with `num_labels=1`. The LM head is stripped and the reward head is renamed to `score` — no custom code needed for inference (see [Section 11](#11-inference-after-training)).

---

## 7. Launch (8 Nodes, 64 GPUs)

The launch script [`launch_reward_training.sh`](./launch_reward_training.sh) must run on **every node** simultaneously, each with a unique `MACHINE_RANK` (0 through 7). Your job scheduler handles this orchestration.

### 7.1 Full Launch Command

```bash
#!/usr/bin/env bash

enroot start --root \
    --rw \
    --mount /mnt/fsx/:/mnt/fsx/ \
    --mount /opt/dlami/nvme/shm/:/dev/shm/ \
    --env CUDA_LAUNCH_BLOCKING=0 \
    --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    --env HF_TOKEN="${HF_TOKEN}" \
    --env MASTER_ADDR="${MASTER_ADDR}" \
    --env MASTER_PORT="${MASTER_PORT}" \
    --env MACHINE_RANK="${MACHINE_RANK}" \
    --env NCCL_DEBUG=INFO \
    --env NCCL_NVLS_ENABLE=0 \
    --env NCCL_P2P_LEVEL=NVL \
    --env NCCL_TIMEOUT=300 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env TOKENIZERS_PARALLELISM=false \
    --env TORCH_DIST_INIT_BARRIER=1 \
    --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
    --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    --env WANDB_API_KEY="${WANDB_API_KEY}" \
    --env WANDB_LOG_MODEL=false \
    Themis accelerate launch \
        --config_file "/mnt/fsx/Themis/v1.1/Standalone_Trainer/fsdp2_config.yaml" \
        --log_dir "/mnt/fsx/Themis/v1.1/27B/Logs/" \
        --machine_rank "${MACHINE_RANK}" \
        --main_process_ip "${MASTER_ADDR}" \
        --main_process_port "${MASTER_PORT}" \
        --multi_gpu \
        --rdzv_backend "c10d" \
        --tee 3 \
        /mnt/fsx/Themis/v1.1/Standalone_Trainer/train_reward_model.py \
            --model_name_or_path "Qwen/Qwen3-14B" \
            --dataset_name "project-themis/Themis-GeneralPreference" \
            --dataset_split "train" \
            --max_length 2560 \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 4 \
            --num_train_epochs 2 \
            --learning_rate 2e-5 \
            --weight_decay 0.01 \
            --warmup_ratio 0.05 \
            --stable_ratio 0.7 \
            --decay_ratio 0.25 \
            --warmup_type "linear" \
            --decay_type "cosine" \
            --min_lr_ratio 0.5 \
            --lambda_lm 0.4 \
            --lambda_mag 0.01 \
            --output_dir "/mnt/fsx/Themis/v1.1/27B/Outputs/" \
            --save_steps 50 \
            --save_epochs \
            --logging_steps 10 \
            --report_to wandb \
            --wandb_project "LibraRM" \
            --wandb_run_name "Qwen3-14B-PMP" \
            --num_proc 16 \
            --seed 42
```

### 7.2 Container Runtime Flags Explained

| Flag | Purpose |
|------|---------|
| `--root` | Run as root inside the container (required for GPU access and shared memory) |
| `--rw` | Writable rootfs — needed for HuggingFace cache, temp files, NCCL shared state |
| `--mount /mnt/fsx/:/mnt/fsx/` | Bind-mount the shared filesystem (shared across all nodes for code, data, checkpoints) |
| `--mount /opt/dlami/nvme/shm/:/dev/shm/` | NVMe-backed `/dev/shm` for NCCL shared memory buffers. The default tmpfs is often too small for large allreduce operations. |

### 7.3 accelerate launch Flags

| Flag | Purpose |
|------|---------|
| `--config_file` | Path to [`fsdp2_config.yaml`](./fsdp2_config.yaml) |
| `--machine_rank` | This node's index in the cluster |
| `--main_process_ip` | IP of rank-0 for rendezvous |
| `--main_process_port` | Port for c10d rendezvous store |
| `--multi_gpu` | Launch one process per GPU (8 per node) |
| `--rdzv_backend "c10d"` | Use PyTorch's native TCP rendezvous |
| `--tee 3` | Duplicate stdout/stderr of all ranks to the log dir |
| `--log_dir` | Where per-rank logs are written |

### 7.4 NCCL / Torch Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `NCCL_NVLS_ENABLE=0` | Disable NVLink SHARP — can cause hangs on heterogeneous topologies |
| `NCCL_P2P_LEVEL=NVL` | Use NVLink for intra-node GPU-to-GPU P2P transfers |
| `NCCL_TIMEOUT=300` | 5-minute timeout for NCCL collectives (increase if checkpointing is slow) |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduce CUDA memory fragmentation by allowing the allocator to expand in-place |
| `TOKENIZERS_PARALLELISM=false` | Avoid deadlocks from forked tokenizer workers |
| `TORCH_DIST_INIT_BARRIER=1` | Synchronize all ranks at process group init (prevents fast ranks from racing ahead) |
| `TORCH_DISTRIBUTED_DEBUG=DETAIL` | Verbose logging of all collective operations (set to `OFF` in production for less noise) |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` | Detect NCCL errors asynchronously and raise them as exceptions instead of hanging |

---

## 8. Monitoring & Debugging

### 8.1 Live GPU Monitoring

SSH into any node and run (inside or outside the container):

```bash
nvitop          # interactive TUI (installed in the container)
nvidia-smi -l 5 # refresh every 5 seconds
```

Key things to watch:
- **GPU Utilization**: should be >85% during forward/backward (dips during checkpointing are normal)
- **Memory usage**: should be close to capacity (gradient checkpointing + FSDP makes this tight)
- **Temperature**: sustained >80C may indicate cooling issues

### 8.2 Weights & Biases

If `--report_to wandb` is set, metrics are logged live:
- `loss/total`, `loss/bt`, `loss/lm`, `loss/mag` — loss components
- `reward/chosen_mean`, `reward/rejected_mean` — mean reward values (watch for divergence)
- `reward/margin` — mean(r_chosen - r_rejected), should increase over training
- `reward/accuracy` — fraction where r_chosen > r_rejected, should approach 0.7-0.8
- `train/learning_rate` — verify WSD schedule looks correct

### 8.3 Per-Rank Logs

Each rank writes logs to `--log_dir`:
```bash
ls /mnt/fsx/Themis/v1.1/27B/Logs/
# main_log.txt, rank_0.log, rank_1.log, ...
```

Check rank-0 for training progress; check other ranks if you suspect NCCL hangs (a stuck rank will stop logging).

### 8.4 NCCL Debugging

If the job hangs:
1. Check that `NCCL_DEBUG=INFO` is set — the logs will show which collective is stuck
2. Look for `NCCL WARN Timeout` in any rank's output
3. Try increasing `NCCL_TIMEOUT` if it only happens during checkpointing (which gathers the full state dict)
4. Verify high-speed networking connectivity: `fi_info -p efa` (or equivalent for your fabric) inside the container should list network devices

---

## 9. Checkpointing & Recovery

### 9.1 Checkpoint Layout

```
/mnt/fsx/Themis/v1.1/27B/Outputs/
  checkpoint-50/          # Step-based checkpoints
    config.json
    model.safetensors
    tokenizer.json
    ...
  checkpoint-100/
  epoch-1/                # Epoch-end checkpoints (if --save_epochs)
  epoch-2/
  config.json             # Final model (saved at end of training)
  model.safetensors
  tokenizer.json
  ...
```

### 9.2 Checkpoint Format

Every checkpoint is a standalone `AutoModelForSequenceClassification` with `num_labels=1`. The training script handles the key remapping:
- Strips the `backbone.` prefix from parameter names
- Drops the `lm_head.*` weights (not needed for reward inference)
- Renames `reward_head.weight` to `score.weight` (HuggingFace convention)

This means checkpoints can be loaded directly for inference or evaluation without any custom model code.

### 9.3 Resuming from Checkpoint

The current training script does not support mid-run resume. If a job fails:
- The last complete checkpoint on disk is a valid model
- Start a new training run from the base model (or fine-tune from the checkpoint as a new `--model_name_or_path`)

### 9.4 Disk Space

Checkpoint sizes scale linearly with model parameters (bf16 safetensors: roughly 2 bytes per parameter). With frequent `--save_steps`, many checkpoints can accumulate. Plan shared filesystem capacity accordingly or increase `--save_steps`.

---

## 10. Adapting for a Different Model

### 10.1 Steps

1. **Update `--model_name_or_path`** in the launch script.

2. **Find the decoder layer class name** for [`fsdp2_config.yaml`](./fsdp2_config.yaml).

   This training script requires [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
   support. The table below lists all compatible model families, their decoder layer
   class (for `fsdp_transformer_layer_cls_to_wrap`), and example HuggingFace model IDs.

   | Model Family | `fsdp_transformer_layer_cls_to_wrap` | Example Model IDs |
   |---|---|---|
   | Llama 2 / 3 / 3.1 / 3.2 | `LlamaDecoderLayer` | `meta-llama/Llama-3.1-8B`, `meta-llama/Llama-3.2-3B` |
   | Llama 4 | `Llama4TextDecoderLayer` | `meta-llama/Llama-4-Scout-17B-16E` |
   | Mistral | `MistralDecoderLayer` | `mistralai/Mistral-7B-v0.3` |
   | Ministral | `MinistralDecoderLayer` | `mistralai/Ministral-8B-Instruct-2410` |
   | Mixtral | `MixtralDecoderLayer` | `mistralai/Mixtral-8x7B-v0.1` |
   | Qwen 2 / 2.5 / QwQ | `Qwen2DecoderLayer` | `Qwen/Qwen2.5-72B`, `Qwen/QwQ-32B` |
   | Qwen 3 | `Qwen3DecoderLayer` | `Qwen/Qwen3-14B`, `Qwen/Qwen3-32B` |
   | Qwen 3 MoE | `Qwen3MoeDecoderLayer` | `Qwen/Qwen3-235B-A22B` |
   | Qwen 3.5 | `Qwen3_5DecoderLayer` | `Qwen/Qwen3.5-27B` |
   | Gemma 1 | `GemmaDecoderLayer` | `google/gemma-7b` |
   | Gemma 2 | `Gemma2DecoderLayer` | `google/gemma-2-27b` |
   | Gemma 3 | `Gemma3DecoderLayer` | `google/gemma-3-27b-pt` |
   | Phi-3 / Phi-3.5 | `Phi3DecoderLayer` | `microsoft/Phi-3.5-mini-instruct` |
   | Granite 3.0 / 3.1 | `GraniteDecoderLayer` | `ibm-granite/granite-3.1-8b-instruct` |
   | Nemotron | `NemotronDecoderLayer` | `nvidia/Nemotron-4-340B-Base` |
   | OLMo 2 | `Olmo2DecoderLayer` | `allenai/OLMo-2-7B` |
   | OLMo 3 | `Olmo3DecoderLayer` | `allenai/OLMo-3-8B` |
   | Falcon H1 | `FalconH1DecoderLayer` | `tiiuae/Falcon-H1-34B-Base` |
   | GLM-4 | `Glm4DecoderLayer` | `THUDM/glm-4-9b` |
   | SmolLM 3 | `SmolLM3DecoderLayer` | `HuggingFaceTB/SmolLM3-3B` |
   | EXAONE 4 | `Exaone4DecoderLayer` | `LGAI-EXAONE/EXAONE-4-32B` |

   **Not supported by Liger** (will fail at startup): DeepSeek-V2/V3, Command-R/R2 (Cohere), Falcon (original), GPT-NeoX/Pythia.

   If unsure, inspect the model programmatically:
   ```python
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("model-name", device_map="meta")
   for name, module in model.named_modules():
       if "layers.0" in name and name.endswith(".0"):
           print(type(module).__name__)
           break
   ```

3. **Adjust `--max_length`** based on the model's context window and your data. Longer sequences need more memory per sample.

4. **Adjust batch size** — for larger models, reduce `--per_device_train_batch_size` to avoid OOM. Compensate with higher `--gradient_accumulation_steps` to maintain effective batch size.

5. **Verify Liger compatibility** — if the model architecture isn't in the table above, `AutoLigerKernelForCausalLM.from_pretrained()` will fail at startup with a clear error.

### 10.2 Memory Budget

Per-device GPU memory depends on model size, FSDP sharding strategy, gradient checkpointing, and batch size. Start with a small `--per_device_train_batch_size` (1-2) and increase until GPU utilisation is satisfactory. Use `reshard_after_forward: true` (ZeRO-3) for tighter memory, or `false` (ZeRO-2) if you have headroom and want higher throughput.

---

## 11. Inference After Training

The saved model is a standard HuggingFace `AutoModelForSequenceClassification`. No custom code or Liger kernel is needed at inference time.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_path = "/mnt/fsx/Themis/v1.1/27B/Outputs/"
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Score a single response
prompt = "Explain quantum entanglement simply."
response = "Quantum entanglement is when two particles..."
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt},
     {"role": "assistant", "content": response}],
    tokenize=False,
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    reward = model(**inputs).logits.squeeze().item()
print(f"Reward score: {reward:.4f}")
```

### Comparing Two Responses (Best-of-N)

```python
def score(prompt, response):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": response}],
        tokenize=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        return model(**inputs).logits.squeeze().item()

r1 = score("What is 2+2?", "The answer is 4.")
r2 = score("What is 2+2?", "I think it might be 5.")
print(f"Response 1: {r1:.4f}, Response 2: {r2:.4f}")
# Response 1 should score higher
```

---

## 12. Troubleshooting

### Job hangs at startup (no training begins)

- **Cause**: Rank-0 rendezvous store is unreachable from other nodes.
- **Fix**: Verify `MASTER_ADDR` is correct and the port is open in security groups. Check that all nodes start within `NCCL_TIMEOUT` seconds of each other.

### CUDA OOM

- Reduce `--per_device_train_batch_size` (compensate with higher `--gradient_accumulation_steps`)
- Ensure `--gradient_checkpointing` is enabled (default: yes)
- Set `fsdp_reshard_after_forward: true` in the YAML
- Reduce `--max_length` (shorter sequences = less activation memory)
- As a last resort: set `fsdp_offload_params: true` (significant speed penalty)

### Reward accuracy stuck at 0.5

- The reward head is zero-initialised, so accuracy starts at ~0.5. If it doesn't improve after 50-100 steps:
- Check that chosen/rejected aren't swapped in your dataset
- Try a lower `--lambda_lm` (too high can dominate the BT signal)
- Verify the data isn't all being truncated (check truncation warnings in logs)

### NaN loss

- Usually caused by reward magnitude explosion. Increase `--lambda_mag` (e.g. 0.05-0.1)
- Can also indicate a learning rate that's too high — try halving `--learning_rate`

### Slow training (low GPU utilization)

- Check `NCCL_DEBUG=INFO` logs for slow collectives (indicates network issues)
- Ensure high-speed networking is working: `fi_info -p efa` (or equivalent) should list devices
- Verify `/dev/shm` is backed by fast storage (not the default 64MB tmpfs)
- Consider `fsdp_reshard_after_forward: false` if you have memory headroom

### Checkpoint save is very slow

- `FULL_STATE_DICT` gathers all parameters to rank-0, which takes time for large models
- For faster saves: switch to `SHARDED_STATE_DICT` in the YAML (but requires the same world size to resume)
- Increase `NCCL_TIMEOUT` if saves trigger timeout errors
