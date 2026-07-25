#!/usr/bin/env bash
# Run Qwen2.5-1.5B baselines in strict priority order. The method loop is
# deliberately outside the dataset loop: a baseline must finish every selected
# dataset before the next baseline can start. Any cell failure stops the queue.
set -euo pipefail

WORKSPACE=$(dirname "$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")")
cd "$WORKSPACE"

export AMO_PY=${AMO_PY:-/home/rihongqiu/data/miniconda3/envs/amo/bin/python}
export TRAIN_GPUS=${TRAIN_GPUS:-2,3}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
NEWS_SERVER_GPUS=${NEWS_SERVER_GPUS:-3}
SAFE_SERVER_GPUS=${SAFE_SERVER_GPUS:-2,3}
export HELPFUL_TARGET_HOST=${HELPFUL_TARGET_HOST:-127.0.0.1}
export HELPFUL_TARGET_PORT=${HELPFUL_TARGET_PORT:-50051}
export HARMLESS_TARGET_HOST=${HARMLESS_TARGET_HOST:-127.0.0.1}
export HARMLESS_TARGET_PORT=${HARMLESS_TARGET_PORT:-50052}
SAFE_HELPFUL_MODEL_PATH=${SAFE_HELPFUL_MODEL_PATH:-$WORKSPACE/playground/reward_model/checkpoints/Qwen2.5-7B-SafeRLHF-RM}
SAFE_HARMLESS_MODEL_PATH=${SAFE_HARMLESS_MODEL_PATH:-$WORKSPACE/playground/reward_model/checkpoints/Qwen2.5-7B-SafeRLHF-CM}
MAX_ACTOR_CKPTS=${MAX_ACTOR_CKPTS:-3}
MODEL=${BASELINE_MODEL:-1.5b}

DEFAULT_METHODS="grpo tchebycheff rvpo ctwa lagrangian fair_stable mgda gapo dynamic_hv nsga2 smsemoa"
DEFAULT_DATASETS="math-lighteval pku-saferlhf rlla"
METHODS_INPUT=${BASELINE_METHODS:-$DEFAULT_METHODS}
DATASETS_INPUT=${BASELINE_DATASETS:-$DEFAULT_DATASETS}
METHODS_INPUT=${METHODS_INPUT//,/ }
DATASETS_INPUT=${DATASETS_INPUT//,/ }
read -r -a METHODS <<< "$METHODS_INPUT"
read -r -a DATASETS <<< "$DATASETS_INPUT"

case "$MODEL" in
    1.5b|qwen2.5-1.5b) MODEL_TAG=qwen2.5-1.5b ;;
    *) echo "priority baseline queue only supports Qwen2.5-1.5B; got: $MODEL" >&2; exit 2 ;;
esac
[[ $MAX_ACTOR_CKPTS =~ ^[1-9][0-9]*$ ]] || { echo "MAX_ACTOR_CKPTS must be positive" >&2; exit 2; }

LOGDIR=$WORKSPACE/train_logs/priority_baselines
MARKER_DIR=$LOGDIR/completed_train
LEDGER=$LOGDIR/queue_progress.log
STATUS_FILE=$LOGDIR/queue.status
NEWS_LOG=$LOGDIR/news_server.log
SAFE_HELPFUL_LOG=$LOGDIR/safe_helpful_server.log
SAFE_HARMLESS_LOG=$LOGDIR/safe_harmless_server.log
mkdir -p "$MARKER_DIR"

exec 9>"$LOGDIR/queue.lock"
if ! flock -n 9; then
    echo "another priority baseline queue already holds $LOGDIR/queue.lock" >&2
    exit 1
fi

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$LEDGER"
}

project_for_dataset() {
    case "$1" in
        math-lighteval) echo amo_math-lighteval ;;
        pku-saferlhf) echo amo_pku-saferlhf ;;
        news) echo amo_cnn_dailymail ;;
        rlla) echo amo_rlla ;;
        *) return 1 ;;
    esac
}

results_dir_for_dataset() {
    case "$1" in
        math-lighteval) echo MATH-LightEval ;;
        pku-saferlhf) echo PKU-SafeRLHF ;;
        news) echo CNN_DailyMail ;;
        rlla) echo RLLA ;;
        *) return 1 ;;
    esac
}

validate_method() {
    case "$1" in
        grpo|tchebycheff|rvpo|ctwa|lagrangian|fair_stable|mgda|gapo|dynamic_hv|nsga2|smsemoa) ;;
        *) echo "unsupported baseline: $1" >&2; return 1 ;;
    esac
}

# Empty is the uniform centroid and intentionally keeps the historical artifact
# name. LS and weighted GDPO share the same non-uniform H=2 simplex grid.
set_cell_variants() {
    local method=$1
    local dataset=$2
    CELL_VARIANTS=("")
    if [[ $method != grpo ]]; then
        return 0
    fi

    case "$dataset" in
        math-lighteval)
            CELL_VARIANTS+=(h2w200 h2w020 h2w002 h2w110 h2w101 h2w011)
            ;;
        news)
            CELL_VARIANTS+=(
                h2w2000 h2w0200 h2w0020 h2w0002
                h2w1100 h2w1010 h2w1001 h2w0110 h2w0101 h2w0011
            )
            ;;
        pku-saferlhf|rlla)
            CELL_VARIANTS+=(h2w20 h2w02)
            ;;
        *)
            echo "no priority weight sweep defined for dataset: $dataset" >&2
            return 1
            ;;
    esac
}

port_ready() {
    local host=$1
    local port=$2
    "$AMO_PY" -c 'import socket,sys; s=socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1); s.close()' \
        "$host" "$port" >/dev/null 2>&1
}

NEWS_SERVER_PID=""
NEWS_SERVER_OWNED=0
ensure_news_server() {
    if port_ready 127.0.0.1 50053; then
        log "NEWS server already ready on 127.0.0.1:50053; reusing it"
        return 0
    fi

    log "NEWS server starting on GPUs $NEWS_SERVER_GPUS -> $NEWS_LOG"
    (
        exec 9>&-
        cd "$WORKSPACE/recipe/amo_news"
        export CUDA_VISIBLE_DEVICES=$NEWS_SERVER_GPUS
        export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
        exec "$AMO_PY" summarization_server.py --port 50053
    ) >> "$NEWS_LOG" 2>&1 &
    NEWS_SERVER_PID=$!
    NEWS_SERVER_OWNED=1

    local attempt
    for ((attempt=1; attempt<=180; attempt++)); do
        if ! kill -0 "$NEWS_SERVER_PID" 2>/dev/null; then
            log "NEWS server exited before becoming ready (see $NEWS_LOG)"
            return 1
        fi
        if port_ready 127.0.0.1 50053; then
            log "NEWS server ready (pid=$NEWS_SERVER_PID)"
            return 0
        fi
        sleep 5
    done
    log "NEWS server did not become ready within 15 minutes"
    return 1
}

SAFE_HELPFUL_SERVER_PID=""
SAFE_HELPFUL_SERVER_OWNED=0
SAFE_HARMLESS_SERVER_PID=""
SAFE_HARMLESS_SERVER_OWNED=0

is_local_host() {
    case "$1" in
        localhost|127.0.0.1|::1) return 0 ;;
        *) return 1 ;;
    esac
}

start_safe_server() {
    local label=$1
    local model_path=$2
    local host=$3
    local port=$4
    local gpu=$5
    local server_log=$6

    is_local_host "$host" || {
        log "$label target $host:$port is remote and not ready; refusing to start it locally"
        return 1
    }
    [[ -d $model_path ]] || {
        log "$label model path is missing: $model_path"
        return 1
    }

    log "$label server starting on GPU $gpu at $host:$port -> $server_log"
    (
        exec 9>&-
        cd "$WORKSPACE/recipe/amo_safe"
        export CUDA_VISIBLE_DEVICES=$gpu
        export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
        exec "$AMO_PY" reward_server.py --model_path "$model_path" --port "$port"
    ) >> "$server_log" 2>&1 &
    STARTED_SAFE_SERVER_PID=$!
}

ensure_safe_servers() {
    local safe_helpful_gpu safe_harmless_gpu safe_extra_gpu
    IFS=',' read -r safe_helpful_gpu safe_harmless_gpu safe_extra_gpu <<< "$SAFE_SERVER_GPUS"
    if [[ -z $safe_helpful_gpu || -z $safe_harmless_gpu || -n ${safe_extra_gpu:-} ]]; then
        log "SAFE_SERVER_GPUS must contain exactly two comma-separated GPU ids"
        return 1
    fi

    local helpful_ready=0
    local harmless_ready=0
    port_ready "$HELPFUL_TARGET_HOST" "$HELPFUL_TARGET_PORT" && helpful_ready=1
    port_ready "$HARMLESS_TARGET_HOST" "$HARMLESS_TARGET_PORT" && harmless_ready=1
    if [[ $helpful_ready == 1 && $harmless_ready == 1 ]]; then
        log "SAFE servers already ready on $HELPFUL_TARGET_HOST:$HELPFUL_TARGET_PORT and $HARMLESS_TARGET_HOST:$HARMLESS_TARGET_PORT; reusing them"
        return 0
    fi

    if [[ $helpful_ready == 0 ]]; then
        start_safe_server helpful "$SAFE_HELPFUL_MODEL_PATH" "$HELPFUL_TARGET_HOST" \
            "$HELPFUL_TARGET_PORT" "$safe_helpful_gpu" "$SAFE_HELPFUL_LOG"
        SAFE_HELPFUL_SERVER_PID=$STARTED_SAFE_SERVER_PID
        SAFE_HELPFUL_SERVER_OWNED=1
    fi
    if [[ $harmless_ready == 0 ]]; then
        start_safe_server harmless "$SAFE_HARMLESS_MODEL_PATH" "$HARMLESS_TARGET_HOST" \
            "$HARMLESS_TARGET_PORT" "$safe_harmless_gpu" "$SAFE_HARMLESS_LOG"
        SAFE_HARMLESS_SERVER_PID=$STARTED_SAFE_SERVER_PID
        SAFE_HARMLESS_SERVER_OWNED=1
    fi

    local attempt
    for ((attempt=1; attempt<=180; attempt++)); do
        if [[ $SAFE_HELPFUL_SERVER_OWNED == 1 ]] && ! kill -0 "$SAFE_HELPFUL_SERVER_PID" 2>/dev/null; then
            log "helpful server exited before becoming ready (see $SAFE_HELPFUL_LOG)"
            return 1
        fi
        if [[ $SAFE_HARMLESS_SERVER_OWNED == 1 ]] && ! kill -0 "$SAFE_HARMLESS_SERVER_PID" 2>/dev/null; then
            log "harmless server exited before becoming ready (see $SAFE_HARMLESS_LOG)"
            return 1
        fi
        if port_ready "$HELPFUL_TARGET_HOST" "$HELPFUL_TARGET_PORT" && \
            port_ready "$HARMLESS_TARGET_HOST" "$HARMLESS_TARGET_PORT"; then
            log "SAFE servers ready (helpful pid=${SAFE_HELPFUL_SERVER_PID:-external}; harmless pid=${SAFE_HARMLESS_SERVER_PID:-external})"
            return 0
        fi
        sleep 5
    done
    log "SAFE servers did not both become ready within 15 minutes"
    return 1
}

stop_owned_server() {
    local owned=$1
    local pid=$2
    local label=$3
    if [[ $owned == 1 && -n $pid ]]; then
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
        log "$label server released (pid=$pid)"
    fi
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    stop_owned_server "$NEWS_SERVER_OWNED" "$NEWS_SERVER_PID" NEWS
    stop_owned_server "$SAFE_HELPFUL_SERVER_OWNED" "$SAFE_HELPFUL_SERVER_PID" helpful
    stop_owned_server "$SAFE_HARMLESS_SERVER_OWNED" "$SAFE_HARMLESS_SERVER_PID" harmless
    printf "finished rc=%s time=%s\n" "$rc" "$(date '+%F %T')" > "$STATUS_FILE"
    exit "$rc"
}
trap cleanup EXIT INT TERM

for method in "${METHODS[@]}"; do
    validate_method "$method"
done
for dataset in "${DATASETS[@]}"; do
    project_for_dataset "$dataset" >/dev/null
done

printf "running pid=%s time=%s\n" "$$" "$(date '+%F %T')" > "$STATUS_FILE"
log "=== priority baseline queue START pid=$$ ==="
log "model: $MODEL_TAG"
log "methods: ${METHODS[*]}"
log "datasets (inside each method): ${DATASETS[*]}"
log "training GPUs: $TRAIN_GPUS; SAFE server GPUs: $SAFE_SERVER_GPUS; NEWS server GPUs: $NEWS_SERVER_GPUS"
log "actor checkpoint retention: $MAX_ACTOR_CKPTS"

for method in "${METHODS[@]}"; do
    log "=== BASELINE START: $method ==="
    for dataset in "${DATASETS[@]}"; do
        entry=$WORKSPACE/scripts/trainers/$method/run_$dataset.sh
        [[ -x $entry ]] || { log "MISSING entry: $entry"; exit 1; }

        project=$(project_for_dataset "$dataset")
        results_dir=$(results_dir_for_dataset "$dataset")
        set_cell_variants "$method" "$dataset"
        for variant in "${CELL_VARIANTS[@]}"; do
            method_tag=$method
            cell_tag=$method.$dataset
            cell_label=$method/$dataset
            marker_variant=base
            if [[ -n $variant ]]; then
                method_tag=${method}_${variant}
                cell_tag=$method.$dataset.$variant
                cell_label=$method/$dataset/$variant
                marker_variant=$variant
            fi

            experiment=${MODEL_TAG}_${method_tag}
            checkpoint_dir=$WORKSPACE/checkpoints/$project/$experiment
            latest=$checkpoint_dir/latest_checkpointed_iteration.txt
            result_json=$WORKSPACE/results/$results_dir/$experiment.json
            marker=$MARKER_DIR/${cell_tag}.done
            train_log=$LOGDIR/${cell_tag}.train.log

            if [[ -s $result_json ]]; then
                log "SKIP $cell_label (canonical result exists)"
                continue
            fi
            if [[ -s $marker && -s $latest ]]; then
                log "SKIP $cell_label (training marker and checkpoint exist)"
                continue
            fi
            if [[ $dataset == pku-saferlhf ]]; then
                ensure_safe_servers
            fi
            if [[ $dataset == news ]]; then
                ensure_news_server
            fi

            log "TRAIN $cell_label -> $train_log"
            if TRAINER_VARIANT="$variant" EXPERIMENT_NAME="$experiment" CHECKPOINT_DIR="$checkpoint_dir" \
                bash "$entry" "$MODEL" "trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPTS" >> "$train_log" 2>&1; then
                printf "completed time=%s method=%s dataset=%s experiment=%s variant=%s checkpoint=%s\n" \
                    "$(date '+%F %T')" "$method" "$dataset" "$experiment" "$marker_variant" "$latest" \
                    > "$marker"
                log "TRAIN ok $cell_label"
            else
                rc=$?
                log "TRAIN FAIL $cell_label rc=$rc (see $train_log); queue stopping"
                exit "$rc"
            fi
        done
    done
    log "=== BASELINE COMPLETE ON ALL DATASETS: $method ==="
done

log "=== priority baseline queue COMPLETE ==="
