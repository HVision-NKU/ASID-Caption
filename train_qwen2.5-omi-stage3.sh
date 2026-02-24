#!/usr/bin/env bash
set -euo pipefail

echo "training start!!!!!!!"

# ------------------------------
# 0) Basic distributed env
# ------------------------------
# These are from your platform (Arnold/Metis). Adjust if your env names differ.
NNODES="${ARNOLD_WORKER_NUM:-1}"          # number of nodes
NODE_RANK="${ARNOLD_ID:-0}"               # this node's rank
GPUS_PER_NODE="${ARNOLD_WORKER_GPU:-1}"   # gpus per node
MASTER_ADDR="${METIS_WORKER_0_HOST:-127.0.0.1}"

# METIS_WORKER_0_PORT may be "29500,29501,..." -> take first
ports=($(echo "${METIS_WORKER_0_PORT:-29500}" | tr ',' ' '))
MASTER_PORT="${ports[0]:-29500}"

echo "NNODES:         ${NNODES}"
echo "NODE_RANK:      ${NODE_RANK}"
echo "GPUS_PER_NODE:  ${GPUS_PER_NODE}"
echo "MASTER_ADDR:    ${MASTER_ADDR}"
echo "MASTER_PORT:    ${MASTER_PORT}"

# ------------------------------
# 1) Disable audio (explicit)
# ------------------------------
export USE_AUDIO_IN_VIDEO=1
export ENABLE_AUDIO_OUTPUT=0

# Optional: avoid noisy warnings
export PYTHONWARNINGS="ignore:PySoundFile failed. Trying audioread instead."

# ------------------------------
# 2) Vision token budget (same as yours)
# ------------------------------
export VIDEO_MAX_PIXELS=200704      
export VIDEO_TOTAL_PIXELS=12042240  


# ------------------------------
# 3) Your training config
# ------------------------------
MODEL="ASID-Caption/output/Qwen2.5-Omni-3B/20260119/"

DATASET="1_2_m_youtube_v0_1/train/all_attributes_1_2_m_youtube_v0_1.jsonl"

OUTDIR="output/Qwen2.5-Omni-3B/"

# Gradient accumulation: keep your original logic
# (You can change 128 to your desired global batch target)
GRAD_ACCUM="$(expr 128 / ${GPUS_PER_NODE} / ${NNODES})"
if [ "${GRAD_ACCUM}" -lt 1 ]; then
  GRAD_ACCUM=1
fi
echo "GRAD_ACCUM:     ${GRAD_ACCUM}"

# ------------------------------
# 4) Launch with torchrun (multi-node multi-gpu)
# ------------------------------
# IMPORTANT:
# - -m swift.cli.sft calls the same entrypoint as `swift sft`
# - torchrun ensures each rank uses its own local GPU correctly
# - This script does NOT use audio (USE_AUDIO_IN_VIDEO=0)
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m swift.cli.sft \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --learning_rate 2e-5 \
    --max_length 22384 \
    --save_steps 60 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 8 \
    --output_dir "${OUTDIR}" \
    --system 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving visual inputs and generating text.' \
    --deepspeed zero3_offload \
    --use_liger_kernel true\
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \