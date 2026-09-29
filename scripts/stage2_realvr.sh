#!/bin/bash
# Stage 2: GRPO with ReaLVR evidence supervision, initialized from a Stage-1 checkpoint.
# Multi-node: set NNODES, NODE_RANK, MASTER_ADDR on every node.
set -euo pipefail

MODEL=${MODEL:-checkpoints/stage1}
DATA=${DATA:-data/stage2.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-data/images}
OUTPUT=${OUTPUT:-checkpoints/stage2}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

torchrun --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$GPUS_PER_NODE" \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  -m realvr.train.train_realvr \
  --deepspeed "$ROOT/scripts/zero3.json" \
  --model_id "$MODEL" \
  --data_path "$DATA" \
  --image_folder "$IMAGE_FOLDER" \
  --image_max_pixels $((2560 * 28 * 28)) \
  --output_dir "$OUTPUT" \
  --bf16 True \
  --gradient_checkpointing True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_generations 8 \
  --temperature 0.6 \
  --top_p 1.0 \
  --top_k 0 \
  --max_completion_length 192 \
  --beta 0.0 \
  --learning_rate 5e-7 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.1 \
  --max_steps 100 \
  --save_steps 25 \
  --logging_steps 1 \
  --lvr_steps 8 \
  --evidence_weight 0.2 \
  --evidence_margin 0.5 \
  --credit_eta 0.3 \
  --num_negatives 16 \
  --report_to none \
  "$@"
