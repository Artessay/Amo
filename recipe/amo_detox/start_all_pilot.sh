#!/usr/bin/env bash

set -euo pipefail

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")
export PYTHONPATH="$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
# Override with HF_HUB_OFFLINE=0 only when the three models need downloading.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

STA_PORT=${DETOX_STA_PORT:-50060}
SIM_PORT=${DETOX_SIM_PORT:-50061}
FL_PORT=${DETOX_FL_PORT:-50062}
STARTUP_TIMEOUT=${DETOX_STARTUP_TIMEOUT:-180}
for port in "$STA_PORT" "$SIM_PORT" "$FL_PORT"; do
    if [[ ! "$port" =~ ^[0-9]+$ ]] || ((10#$port < 1 || 10#$port > 65535)); then
        echo "Reward server ports must be integers between 1 and 65535: $port" >&2
        exit 2
    fi
done
if [[ ! "$STARTUP_TIMEOUT" =~ ^[0-9]+$ ]] || ((10#$STARTUP_TIMEOUT < 1)); then
    echo "DETOX_STARTUP_TIMEOUT must be a positive integer: $STARTUP_TIMEOUT" >&2
    exit 2
fi

pids=()
cleanup() {
    if ((${#pids[@]})); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
    local label=$1
    local port=$2
    local pid=$3
    local deadline=$((SECONDS + STARTUP_TIMEOUT))
    while ((SECONDS < deadline)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$label reward server exited during startup" >&2
            return 1
        fi
        if (exec 3<>/dev/tcp/127.0.0.1/"$port") 2>/dev/null; then
            echo "$label reward server ready on :$port"
            return 0
        fi
        sleep 1
    done
    echo "Timed out after ${STARTUP_TIMEOUT}s waiting for $label reward server on :$port" >&2
    return 1
}

python3 "$WORKSPACE/recipe/amo_detox/offline_server.py" sta \
    --model_path "${DETOX_STA_MODEL_PATH:-s-nlp/roberta_toxicity_classifier}" \
    --port "$STA_PORT" &
pids+=("$!")
python3 "$WORKSPACE/recipe/amo_detox/offline_server.py" sim \
    --model_path "${DETOX_SIM_MODEL_PATH:-sentence-transformers/LaBSE}" \
    --port "$SIM_PORT" &
pids+=("$!")
python3 "$WORKSPACE/recipe/amo_detox/offline_server.py" fl \
    --model_path "${DETOX_FL_MODEL_PATH:-textattack/roberta-base-CoLA}" \
    --port "$FL_PORT" &
pids+=("$!")

wait_for_server STA "$STA_PORT" "${pids[0]}"
wait_for_server SIM "$SIM_PORT" "${pids[1]}"
wait_for_server FL "$FL_PORT" "${pids[2]}"
echo "All ParaDetox reward servers are ready (PIDs: ${pids[*]})"
wait -n "${pids[@]}"
