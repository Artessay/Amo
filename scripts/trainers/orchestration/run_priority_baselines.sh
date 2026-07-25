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
NEWS_SERVER_GPUS=${NEWS_SERVER_GPUS:-0,1}
MAX_ACTOR_CKPTS=${MAX_ACTOR_CKPTS:-3}
MODEL=${BASELINE_MODEL:-1.5b}

DEFAULT_METHODS="ls tchebycheff gdpo_weighted rvpo ctwa lagrangian fair_stable mgda gapo dynamic_hv nsga2 smsemoa"
DEFAULT_DATASETS="math-lighteval news rlla"
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
        news) echo amo_cnn_dailymail ;;
        rlla) echo amo_rlla ;;
        *) return 1 ;;
    esac
}

results_dir_for_dataset() {
    case "$1" in
        math-lighteval) echo MATH-LightEval ;;
        news) echo CNN_DailyMail ;;
        rlla) echo RLLA ;;
        *) return 1 ;;
    esac
}

validate_method() {
    case "$1" in
        ls|tchebycheff|gdpo_weighted|rvpo|ctwa|lagrangian|fair_stable|mgda|gapo|dynamic_hv|nsga2|smsemoa) ;;
        *) echo "unsupported baseline: $1" >&2; return 1 ;;
    esac
}

# Empty is the uniform centroid and intentionally keeps the historical artifact
# name. LS and weighted GDPO share the same non-uniform H=2 simplex grid.
set_cell_variants() {
    local method=$1
    local dataset=$2
    CELL_VARIANTS=("")
    if [[ $method != ls && $method != gdpo_weighted ]]; then
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
        rlla)
            CELL_VARIANTS+=(h2w20 h2w02)
            ;;
        *)
            echo "no priority weight sweep defined for dataset: $dataset" >&2
            return 1
            ;;
    esac
}

port_ready() {
    "$AMO_PY" -c 'import socket; s=socket.create_connection(("127.0.0.1", 50053), timeout=1); s.close()' >/dev/null 2>&1
}

NEWS_SERVER_PID=""
NEWS_SERVER_OWNED=0
ensure_news_server() {
    if port_ready; then
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
        if port_ready; then
            log "NEWS server ready (pid=$NEWS_SERVER_PID)"
            return 0
        fi
        sleep 5
    done
    log "NEWS server did not become ready within 15 minutes"
    return 1
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if [[ $NEWS_SERVER_OWNED == 1 && -n $NEWS_SERVER_PID ]] && kill -0 "$NEWS_SERVER_PID" 2>/dev/null; then
        kill "$NEWS_SERVER_PID" 2>/dev/null || true
        wait "$NEWS_SERVER_PID" 2>/dev/null || true
        log "NEWS server stopped (pid=$NEWS_SERVER_PID)"
    fi
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
log "training GPUs: $TRAIN_GPUS; NEWS server GPUs: $NEWS_SERVER_GPUS"
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
