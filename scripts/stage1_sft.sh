#!/bin/bash
# Stage 1: SFT with latent visual reconstruction, starting from Qwen2.5-VL-7B-Instruct.
# Multi-node: set NNODES, NODE_RANK, MASTER_ADDR on every node.
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
DATA=${DATA:-data/stage1.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-data/images}
OUTPUT=${OUTPUT:-checkpoints/stage1}
MAX_STEPS=${MAX_STEPS:-2500}
MAX_PACKED_TOKENS=${MAX_PACKED_TOKENS:-16384}
MIN_VISUAL_TOKENS=${MIN_VISUAL_TOKENS:-128}
MAX_VISUAL_TOKENS=${MAX_VISUAL_TOKENS:-5120}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

torchrun --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$GPUS_PER_NODE" \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  -m realvr.train.train_sft \
  --deepspeed "$ROOT/scripts/zero3.json" \
  --model_id "$MODEL" \
  --data_path "$DATA" \
  --image_folder "$IMAGE_FOLDER" \
  --image_min_pixels $((MIN_VISUAL_TOKENS * 28 * 28)) \
  --image_max_pixels $((MAX_VISUAL_TOKENS * 28 * 28)) \
  --max_packed_tokens "$MAX_PACKED_TOKENS" \
  --output_dir "$OUTPUT" \
  --bf16 True \
  --gradient_checkpointing True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.1 \
  --loss_lvr_lambda 0.1 \
  --max_steps "$MAX_STEPS" \
  --save_steps 500 \
  --save_total_limit 10 \
  --logging_steps 1 \
  --dataloader_num_workers 8 \
  --report_to none \
  "$@"
