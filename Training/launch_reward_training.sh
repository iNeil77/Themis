#!/usr/bin/env bash
# ==============================================================================
# Reward Model Training Launch Script
# ==============================================================================
#
# Launches distributed reward model training inside an enroot container across
# multiple GPUs on a multi-GPU node.
#
# Prerequisites:
#   1. The enroot container "Themis" must already exist (see README.md for
#      import/create instructions from ineil77/themis:29042026-3).
#   2. Environment variables must be set (see "Required Environment" below).
#   3. The shared filesystem (/mnt/fsx/) must be mounted and contain the
#      training script, accelerate config, and dataset cache.
#
# Required Environment Variables:
#   HF_TOKEN        - HuggingFace API token for model/dataset downloads
#   WANDB_API_KEY   - Weights & Biases API key for experiment tracking
#   MASTER_ADDR     - IP address of the rank-0 node (for torch distributed)
#   MASTER_PORT     - Port for distributed communication (e.g. 29500)
#   MACHINE_RANK    - This node's rank in the cluster (0 to num_machines-1)
#
# Multi-node usage:
#   This script must run on EVERY node in the cluster simultaneously, each
#   with a unique MACHINE_RANK. Use your job scheduler (Slurm, etc.)
#   to orchestrate this.
#
# What this script does:
#   1. Starts the Themis enroot container with root access and read-write rootfs
#   2. Mounts the shared filesystem and NVMe-backed /dev/shm for NCCL buffers
#   3. Passes all necessary environment variables (CUDA, NCCL, torch distributed)
#   4. Inside the container: runs `accelerate launch` with the FSDP2 config,
#      which spawns 8 training processes (one per GPU) that coordinate across nodes
#   5. Training writes checkpoints and logs to the specified output directory on the shared filesystem
#
# ==============================================================================

enroot start --root \
    --rw \
    --mount /mnt/fsx/:/mnt/fsx/ \
    --mount /opt/dlami/nvme/shm/:/dev/shm/ \
    --mount /dev/infiniband/:/dev/infiniband/ \
    \
    `# ---- CUDA configuration ----` \
    --env CUDA_LAUNCH_BLOCKING=0 \
    --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    \
    `# ---- Authentication tokens (read from host environment) ----` \
    --env HF_TOKEN="${HF_TOKEN}" \
    --env WANDB_API_KEY="${WANDB_API_KEY}" \
    --env WANDB_LOG_MODEL=false \
    \
    `# ---- Torch distributed coordination ----` \
    --env MASTER_ADDR="${MASTER_ADDR}" \
    --env MASTER_PORT="${MASTER_PORT}" \
    --env MACHINE_RANK="${MACHINE_RANK}" \
    \
    `# ---- EFA/libfabric device configuration ----` \
    --env FI_EFA_USE_DEVICE_RDMA=1 \
    --env FI_PROVIDER=efa \
    --env FI_EFA_FORK_SAFE=1 \
    \
    `# ---- NCCL tuning for EFA + NVLink topology ----` \
    --env NCCL_DEBUG=INFO \
    --env NCCL_NVLS_ENABLE=0 \
    --env NCCL_P2P_LEVEL=NVL \
    --env NCCL_TIMEOUT=300 \
    \
    `# ---- PyTorch memory and distributed settings ----` \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env TOKENIZERS_PARALLELISM=false \
    --env TORCH_DIST_INIT_BARRIER=1 \
    --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
    --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    \
    `# ---- Container name and accelerate launch command ----` \
    Themis accelerate launch \
        --config_file "/mnt/fsx/Themis/v1.1/Standalone_Trainer/fsdp2_config.yaml" \
        --log_dir "/mnt/fsx/Themis/v1.1/14B/Logs/" \
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
            --output_dir "/mnt/fsx/Themis/v1.1/14B/Outputs/" \
            --save_steps 50 \
            --save_epochs \
            --logging_steps 10 \
            --report_to wandb \
            --wandb_project "ThemisRM" \
            --wandb_run_name "Qwen3-14B-PMP" \
            --num_proc 16 \
            --seed 42
