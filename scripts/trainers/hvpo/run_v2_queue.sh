#!/usr/bin/env bash
set -euo pipefail

# Run the paper-faithful HVPO experiment matrix without ever reusing legacy
# `*_hvpo` checkpoints.  Training uses GPU 2-3; external reward services use
# GPU 0-1.  Every command is executed from the Amo conda environment.

SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
WORKSPACE=$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")
AMO_ENV=${AMO_ENV:-/home/rihongqiu/data/miniconda3/envs/amo}
AMO_PY=${AMO_PY:-$AMO_ENV/bin/python}
TRAIN_GPUS=${TRAIN_GPUS:-2,3}
SERVICE_GPU_HELPFUL=${SERVICE_GPU_HELPFUL:-0}
SERVICE_GPU_HARMLESS=${SERVICE_GPU_HARMLESS:-1}
SERVICE_GPU_NEWS=${SERVICE_GPU_NEWS:-0}
LOG_ROOT=${LOG_ROOT:-$WORKSPACE/logs/hvpo_v2}
STATUS_DIR=$LOG_ROOT/status

mkdir -p "$LOG_ROOT" "$STATUS_DIR"
export AMO_PY TRAIN_GPUS
export NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-2}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
export PATH="$AMO_ENV/bin:$PATH"

declare -a SERVICE_PIDS=()

cleanup_services() {
    local pid
    for pid in "${SERVICE_PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup_services EXIT INT TERM

wait_port() {
    local port=$1
    local service_pid=$2
    local deadline=$((SECONDS + 600))
    while ((SECONDS < deadline)); do
        kill -0 "$service_pid" 2>/dev/null || return 1
        if "$AMO_PY" -c 'import socket,sys; s=socket.socket(); s.settimeout(1); rc=s.connect_ex(("127.0.0.1",int(sys.argv[1]))); s.close(); raise SystemExit(rc)' "$port"; then
            return 0
        fi
        sleep 5
    done
    return 1
}

start_safe_services() {
    if [[ -f $STATUS_DIR/safe_services.ready ]]; then
        return 0
    fi
    CUDA_VISIBLE_DEVICES=$SERVICE_GPU_HELPFUL bash "$WORKSPACE/recipe/amo_safe/start_helpful.sh" \
        >"$LOG_ROOT/safe_helpful.log" 2>&1 &
    local helpful_pid=$!
    SERVICE_PIDS+=("$helpful_pid")
    CUDA_VISIBLE_DEVICES=$SERVICE_GPU_HARMLESS bash "$WORKSPACE/recipe/amo_safe/start_harmless.sh" \
        >"$LOG_ROOT/safe_harmless.log" 2>&1 &
    local harmless_pid=$!
    SERVICE_PIDS+=("$harmless_pid")
    wait_port 50051 "$helpful_pid" || { tail -n 80 "$LOG_ROOT/safe_helpful.log"; return 1; }
    wait_port 50052 "$harmless_pid" || { tail -n 80 "$LOG_ROOT/safe_harmless.log"; return 1; }
    touch "$STATUS_DIR/safe_services.ready"
}

stop_safe_services() {
    cleanup_services
    SERVICE_PIDS=()
    rm -f "$STATUS_DIR/safe_services.ready"
}

start_news_service() {
    CUDA_VISIBLE_DEVICES=$SERVICE_GPU_NEWS bash "$WORKSPACE/recipe/amo_news/start_server.sh" \
        >"$LOG_ROOT/news_service.log" 2>&1 &
    local news_pid=$!
    SERVICE_PIDS+=("$news_pid")
    wait_port 50053 "$news_pid" || { tail -n 80 "$LOG_ROOT/news_service.log"; return 1; }
}

run_one() {
    local dataset=$1
    local model=$2
    local model_tag
    case "$model" in
        1.5b) model_tag=qwen2.5-1.5b ;;
        3b) model_tag=qwen2.5-3b ;;
        *) echo "unsupported model: $model" >&2; return 2 ;;
    esac
    local experiment=${model_tag}_hvpo_v2
    local marker=$STATUS_DIR/${dataset}_${model_tag}.done
    local failure=$STATUS_DIR/${dataset}_${model_tag}.failed
    local log=$LOG_ROOT/${dataset}_${model_tag}.log
    if [[ -f $marker ]]; then
        echo "[queue] already complete: $dataset $model_tag"
        return 0
    fi
    rm -f "$failure"
    echo "[queue] starting: $dataset $model_tag"
    if EXPERIMENT_NAME=$experiment RESUME_MODE=auto \
        bash "$SCRIPT_DIR/run_${dataset}.sh" "$model" >"$log" 2>&1; then
        touch "$marker"
        echo "[queue] complete: $dataset $model_tag"
    else
        local rc=$?
        printf '%s\n' "$rc" >"$failure"
        echo "[queue] failed ($rc): $dataset $model_tag; see $log" >&2
        return "$rc"
    fi
}

run_model_stage() {
    local model=$1
    run_one rlla "$model"
    run_one math-lighteval "$model"
    start_safe_services
    run_one pku-saferlhf "$model"
    stop_safe_services
    start_news_service
    run_one news "$model"
    cleanup_services
    SERVICE_PIDS=()
}

echo "[queue] Amo Python: $AMO_PY"
echo "[queue] training GPUs: $TRAIN_GPUS"
echo "[queue] logs: $LOG_ROOT"
run_model_stage 1.5b
run_model_stage 3b
touch "$STATUS_DIR/ALL_DONE"
echo "[queue] all eight HVPO v2 experiments completed"
