#!/usr/bin/env bash
# Merge, generate, and evaluate one completed trainer cell.
#
# Usage:
#   POSTPROCESS_GPUS=4,5 bash postprocess_cell.sh DATASET EXPERIMENT
#
# DATASET: math-lighteval | news | pku-saferlhf | rlla
# The script is idempotent: a valid merge, parquet, or JSON result is reused.
set -euo pipefail

DATASET_SLUG=${1:?need dataset slug}
EXP=${2:?need experiment name}
WORKSPACE=$(dirname "$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")")
PY=${AMO_PY:-python3}
POSTPROCESS_GPUS=${POSTPROCESS_GPUS:-}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.8}
POSTPROCESS_SERVER_GPUS=${POSTPROCESS_SERVER_GPUS:-}
SERVICE_LOGDIR=$WORKSPACE/train_logs/postprocess/services
mkdir -p "$SERVICE_LOGDIR"

OWNED_SERVICE_PIDS=()
cleanup() {
    local pid
    for pid in "${OWNED_SERVICE_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

port_ready() {
    "$PY" -c 'import socket,sys; s=socket.create_connection((sys.argv[1],int(sys.argv[2])),timeout=1); s.close()' \
        "$1" "$2" >/dev/null 2>&1
}

wait_for_port() {
    local host=$1 port=$2 pid=$3 label=$4 attempt
    for ((attempt=1; attempt<=180; attempt++)); do
        kill -0 "$pid" 2>/dev/null || {
            echo "$label exited before becoming ready" >&2
            return 1
        }
        port_ready "$host" "$port" && return 0
        sleep 5
    done
    echo "$label did not become ready within 15 minutes" >&2
    return 1
}

ensure_reward_services() {
    case "$DATASET_SLUG" in
        news)
            port_ready 127.0.0.1 50053 && return 0
            [[ -n $POSTPROCESS_SERVER_GPUS ]] || {
                echo "POSTPROCESS_SERVER_GPUS is required to start the News reward server" >&2
                return 2
            }
            (
                cd "$WORKSPACE/recipe/amo_news"
                export CUDA_VISIBLE_DEVICES=$POSTPROCESS_SERVER_GPUS
                export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
                exec "$PY" summarization_server.py --port 50053
            ) >>"$SERVICE_LOGDIR/news.log" 2>&1 &
            local news_pid=$!
            OWNED_SERVICE_PIDS+=("$news_pid")
            wait_for_port 127.0.0.1 50053 "$news_pid" "News reward server"
            ;;
        pku-saferlhf)
            if port_ready 127.0.0.1 50051 && port_ready 127.0.0.1 50052; then
                return 0
            fi
            local helpful_gpu harmless_gpu extra_gpu
            IFS=',' read -r helpful_gpu harmless_gpu extra_gpu <<<"$POSTPROCESS_SERVER_GPUS"
            [[ -n $helpful_gpu && -n $harmless_gpu && -z ${extra_gpu:-} ]] || {
                echo "POSTPROCESS_SERVER_GPUS must contain two GPU ids for Safe evaluation" >&2
                return 2
            }
            local helpful_model=$WORKSPACE/playground/reward_model/checkpoints/Qwen2.5-7B-SafeRLHF-RM
            local harmless_model=$WORKSPACE/playground/reward_model/checkpoints/Qwen2.5-7B-SafeRLHF-CM
            if ! port_ready 127.0.0.1 50051; then
                (
                    cd "$WORKSPACE/recipe/amo_safe"
                    export CUDA_VISIBLE_DEVICES=$helpful_gpu
                    export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
                    exec "$PY" reward_server.py --model_path "$helpful_model" --port 50051
                ) >>"$SERVICE_LOGDIR/safe_helpful.log" 2>&1 &
                local helpful_pid=$!
                OWNED_SERVICE_PIDS+=("$helpful_pid")
                wait_for_port 127.0.0.1 50051 "$helpful_pid" "Safe helpful server"
            fi
            if ! port_ready 127.0.0.1 50052; then
                (
                    cd "$WORKSPACE/recipe/amo_safe"
                    export CUDA_VISIBLE_DEVICES=$harmless_gpu
                    export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
                    exec "$PY" reward_server.py --model_path "$harmless_model" --port 50052
                ) >>"$SERVICE_LOGDIR/safe_harmless.log" 2>&1 &
                local harmless_pid=$!
                OWNED_SERVICE_PIDS+=("$harmless_pid")
                wait_for_port 127.0.0.1 50052 "$harmless_pid" "Safe harmless server"
            fi
            ;;
    esac
}

[[ $EXP =~ ^(qwen2\.5-(1\.5b|3b))_[A-Za-z0-9._-]+$ ]] || {
    echo "unsupported experiment name: $EXP" >&2
    exit 2
}

case "$EXP" in
    qwen2.5-1.5b_*) BASE_MODEL=${AMO_MODEL_PATH:-/data/Qwen/Qwen2.5-1.5B-Instruct} ;;
    qwen2.5-3b_*) BASE_MODEL=${AMO_MODEL_PATH:-/data/Qwen/Qwen2.5-3B-Instruct} ;;
esac

case "$DATASET_SLUG" in
    math-lighteval)
        PROJECT=amo_math-lighteval
        DATASET=MATH-LightEval
        MAX_TOKENS=2048
        REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_math/math_accuracy.py','$WORKSPACE/recipe/amo_math/math_conciseness.py','$WORKSPACE/recipe/amo_math/math_format.py']"
        EVAL_OVERRIDES=()
        ;;
    news)
        PROJECT=amo_cnn_dailymail
        DATASET=CNN_DailyMail
        MAX_TOKENS=1024
        REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_news/news_coherence.py','$WORKSPACE/recipe/amo_news/news_fluency.py','$WORKSPACE/recipe/amo_news/news_relevance.py','$WORKSPACE/recipe/amo_news/news_consistency.py']"
        EVAL_OVERRIDES=(data.reward_model_key=extra_info)
        ;;
    pku-saferlhf)
        PROJECT=amo_pku-saferlhf
        DATASET=PKU-SafeRLHF
        MAX_TOKENS=512
        REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_safe/safe_helpfulness.py','$WORKSPACE/recipe/amo_safe/safe_harmlessness.py']"
        EVAL_OVERRIDES=(
            data.reward_model_key=extra_info
            "metrics.calibration_path=$WORKSPACE/results/PKU-SafeRLHF/safe_calibration.json"
        )
        ;;
    rlla)
        PROJECT=amo_rlla
        DATASET=RLLA
        MAX_TOKENS=1024
        REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_tool/tool_correctness.py','$WORKSPACE/recipe/amo_tool/tool_format.py']"
        EVAL_OVERRIDES=()
        ;;
    *)
        echo "unsupported dataset slug: $DATASET_SLUG" >&2
        exit 2
        ;;
esac

CKDIR=$WORKSPACE/checkpoints/$PROJECT/$EXP
LATEST=$CKDIR/latest_checkpointed_iteration.txt
[[ -s $LATEST ]] || { echo "missing checkpoint pointer: $LATEST" >&2; exit 3; }
STEP=$(<"$LATEST")
[[ $STEP =~ ^[1-9][0-9]*$ ]] || { echo "invalid checkpoint step: $STEP" >&2; exit 3; }
ACTOR=$CKDIR/global_step_$STEP/actor
[[ -d $ACTOR ]] || { echo "missing actor checkpoint: $ACTOR" >&2; exit 3; }
if [[ -e $CKDIR/global_step_$STEP/.checkpoint_complete ]]; then
    :
else
    echo "warning: checkpoint lacks .checkpoint_complete: $CKDIR/global_step_$STEP" >&2
fi

ADAPTER=$ACTOR/lora_adapter
MERGE=$ACTOR/merge
DATA=$WORKSPACE/data/$DATASET/test.parquet
OUT=$WORKSPACE/results/$DATASET/$EXP.parquet
JSON=${OUT%.parquet}.json
LOG_PREFIX="[postprocess][$DATASET_SLUG][$EXP]"

[[ -d $BASE_MODEL ]] || { echo "missing base model: $BASE_MODEL" >&2; exit 3; }
[[ -f $DATA ]] || { echo "missing test data: $DATA" >&2; exit 3; }
mkdir -p "$(dirname "$OUT")"

if [[ ! -s $MERGE/config.json ]]; then
    echo "$LOG_PREFIX merging checkpoint step $STEP"
    if [[ -d $ADAPTER ]]; then
        "$PY" "$WORKSPACE/playground/lora_merger.py" \
            --model_path "$BASE_MODEL" --adapter_path "$ADAPTER" --save_path "$MERGE"
    else
        "$PY" "$WORKSPACE/playground/legacy_model_merger.py" merge \
            --backend fsdp --local_dir "$ACTOR" --target_dir "$MERGE"
    fi
else
    echo "$LOG_PREFIX merge already complete"
fi

if [[ ! -s $OUT ]]; then
    [[ -n $POSTPROCESS_GPUS ]] || {
        echo "POSTPROCESS_GPUS is required for inference" >&2
        exit 2
    }
    echo "$LOG_PREFIX generating $OUT on GPUs $POSTPROCESS_GPUS"
    CUDA_VISIBLE_DEVICES=$POSTPROCESS_GPUS XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 \
        "$PY" "$WORKSPACE/playground/generation.py" \
        --model "$MERGE" --data "$DATA" --output "$OUT" \
        --max_tokens "$MAX_TOKENS" --gpu_mem_util "$GPU_MEM_UTIL"
else
    echo "$LOG_PREFIX inference output already exists"
fi

if [[ ! -s $JSON ]]; then
    ensure_reward_services
    echo "$LOG_PREFIX evaluating $OUT"
    XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 "$PY" -m verl.trainer.amo_eval \
        "data.path=$OUT" \
        "custom_reward_function.path=$REWARD_FUNCTION_PATH" \
        "${EVAL_OVERRIDES[@]}"
else
    echo "$LOG_PREFIX evaluation result already exists"
fi

[[ -s $MERGE/config.json && -s $OUT && -s $JSON ]] || {
    echo "$LOG_PREFIX incomplete outputs" >&2
    exit 1
}
echo "$LOG_PREFIX complete: step=$STEP parquet=$OUT json=$JSON"
