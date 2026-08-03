#!/bin/bash
# [Amo] Offline evaluation for one PKU-SafeRLHF experiment:
#   merge LoRA -> generate responses -> amo_eval (means + rooted singleton HV).
#
# This pins all work to GPU 0,1 and lowers vLLM gpu_memory_utilization so
# generation can coexist with the reward gRPC servers already resident on
# GPU 0,1. amo_eval reuses those same servers for scoring.
#
# Writes results/PKU-SafeRLHF/<EXP>.parquet and <EXP>.json.
#
# Usage:
#   bash scripts/trainers/tools/eval_safe.sh <EXP> [MODEL]
#     EXP   : experiment name, e.g. qwen2.5-1.5b_grpo
#     MODEL : 1.5b | 3b | llama3b (inferred from EXP when omitted)
set -x
set -e

EXP=${1:?need EXPERIMENT_NAME e.g. qwen2.5-1.5b_grpo}
MODEL=${2:-}

PROJECT=amo_pku-saferlhf
DATASET=PKU-SafeRLHF
WORKSPACE=$(dirname "$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")")
PY=${AMO_PY:-/home/rihongqiu/data/miniconda3/envs/amo/bin/python}

# Infer base model from the experiment-name prefix if MODEL is not given.
if [ -z "$MODEL" ]; then
  case "$EXP" in
    qwen2.5-1.5b_*) MODEL=1.5b ;;
    qwen2.5-3b_*)   MODEL=3b ;;
    llama3.2-3b_*)  MODEL=llama3b ;;
    *) echo "cannot infer MODEL from EXP=$EXP; pass it explicitly"; exit 1 ;;
  esac
fi
case "$MODEL" in
  1.5b)    BASE_MODEL="/data/Qwen/Qwen2.5-1.5B-Instruct" ;;
  3b)      BASE_MODEL="/data/Qwen/Qwen2.5-3B-Instruct" ;;
  llama3b) BASE_MODEL="/data/meta-llama/Llama-3.2-3B-Instruct" ;;
  *) echo "bad MODEL $MODEL"; exit 1 ;;
esac

# Only GPU 0,1 (shared with reward servers). Small vLLM footprint to fit.
export CUDA_VISIBLE_DEVICES=${EVAL_GPUS:-0,1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.35}

CKDIR=$WORKSPACE/checkpoints/$PROJECT/$EXP
if [ ! -f "$CKDIR/latest_checkpointed_iteration.txt" ]; then
  echo "[eval] no checkpoint for $EXP at $CKDIR -- skipping"; exit 3
fi
STEP=$(cat "$CKDIR/latest_checkpointed_iteration.txt")
ADAPTER=$CKDIR/global_step_$STEP/actor/lora_adapter
MERGE=$CKDIR/global_step_$STEP/actor/merge
# Full test set (8211 prompts), matching the existing GRPO/HVPO result JSONs.
DATA=${EVAL_DATA:-$WORKSPACE/data/$DATASET/test.parquet}
OUT=$WORKSPACE/results/$DATASET/${EXP}.parquet
REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_safe/safe_helpfulness.py','$WORKSPACE/recipe/amo_safe/safe_harmlessness.py']"

mkdir -p "$WORKSPACE/results/$DATASET"

# 1) Merge LoRA or FSDP actor checkpoint -> full HF model (idempotent).
if [ ! -f "$MERGE/config.json" ]; then
  if [ -d "$ADAPTER" ]; then
    echo "[eval] merging LoRA adapter for $EXP (step $STEP)"
    "$PY" "$WORKSPACE/playground/lora_merger.py" \
        --model_path "$BASE_MODEL" --adapter_path "$ADAPTER" --save_path "$MERGE"
  else
    echo "[eval] merging FSDP actor for $EXP (step $STEP)"
    "$PY" "$WORKSPACE/playground/legacy_model_merger.py" merge \
        --backend fsdp \
        --local_dir "$CKDIR/global_step_$STEP/actor" \
        --target_dir "$MERGE"
  fi
else
  echo "[eval] merge exists: $MERGE"
fi

# 2) Generate responses on GPU 0,1 (small vLLM footprint).
echo "[eval] generating responses -> $OUT"
XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 "$PY" "$WORKSPACE/playground/generation.py" \
    --model "$MERGE" --data "$DATA" --output "$OUT" --max_tokens 512 --gpu_mem_util "$GPU_MEM_UTIL"

# 3) Reuse reward servers and the frozen calibration for rooted-HV scoring.
echo "[eval] scoring + rooted hypervolume -> ${OUT%.parquet}.json"
XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 "$PY" -m verl.trainer.amo_eval \
    data.path="$OUT" \
    data.reward_model_key=extra_info \
    metrics.calibration_path="$WORKSPACE/results/$DATASET/safe_calibration.json" \
    custom_reward_function.path="$REWARD_FUNCTION_PATH"
