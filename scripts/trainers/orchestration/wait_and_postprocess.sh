#!/usr/bin/env bash
# Wait for the canonical Qwen2.5 priority matrix to finish, then run the
# resumable merge/inference/evaluation pipeline. A cell is complete only when
# it has a canonical result JSON or a successful training completion marker.
set -euo pipefail

WORKSPACE=$(dirname "$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")")
LOGDIR=$WORKSPACE/train_logs/postprocess
WATCH_LOG=$LOGDIR/watcher.log
INTERVAL=${WATCH_INTERVAL_SECONDS:-300}
STABLE_POLLS=${WATCH_STABLE_POLLS:-2}
export POSTPROCESS_GPUS=${POSTPROCESS_GPUS:-4,5}
export POSTPROCESS_SERVER_GPUS=${POSTPROCESS_SERVER_GPUS:-6,7}
export AMO_PY=${AMO_PY:-/data2/home/qrh/miniconda3/envs/amo/bin/python}
mkdir -p "$LOGDIR"

exec 9>"$LOGDIR/watcher.lock"
flock -n 9 || { echo "another postprocess watcher is already running" >&2; exit 1; }

MODELS=(qwen2.5-1.5b qwen2.5-3b)
DATASETS=(math-lighteval news pku-saferlhf rlla)
METHODS=(dynamic_hv grpo gdpo rvpo tchebycheff ctwa lagrangian fair_stable mgda gapo nsga2 smsemoa)
EXPECTED=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#METHODS[@]}))

result_path() {
    local dataset=$1 experiment=$2 result_dir
    case "$dataset" in
        math-lighteval) result_dir=MATH-LightEval ;;
        news) result_dir=CNN_DailyMail ;;
        pku-saferlhf) result_dir=PKU-SafeRLHF ;;
        rlla) result_dir=RLLA ;;
    esac
    echo "$WORKSPACE/results/$result_dir/$experiment.json"
}

declare -A COMPLETED_MARKERS=()
refresh_markers() {
    COMPLETED_MARKERS=()
    local marker line dataset experiment field
    while IFS= read -r marker; do
        line=$(<"$marker")
        [[ $line == completed\ * ]] || continue
        dataset=""
        experiment=""
        for field in $line; do
            case "$field" in
                dataset=*) dataset=${field#dataset=} ;;
                experiment=*) experiment=${field#experiment=} ;;
            esac
        done
        [[ -n $dataset && -n $experiment ]] || continue
        COMPLETED_MARKERS[$dataset/$experiment]=1
    done < <(find "$WORKSPACE/train_logs/priority_baselines" -type f \
        -path '*/completed_train/*.done' -print)
}

count_complete() {
    refresh_markers
    local count=0 model dataset method experiment json
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for method in "${METHODS[@]}"; do
                experiment=${model}_${method}
                json=$(result_path "$dataset" "$experiment")
                if [[ -s $json || -n ${COMPLETED_MARKERS[$dataset/$experiment]:-} ]]; then
                    count=$((count + 1))
                fi
            done
        done
    done
    echo "$count"
}

stable=0
echo "[$(date '+%F %T')] watcher start expected=$EXPECTED inference_gpus=$POSTPROCESS_GPUS server_gpus=$POSTPROCESS_SERVER_GPUS" \
    >>"$WATCH_LOG"
while :; do
    complete=$(count_complete)
    echo "[$(date '+%F %T')] training matrix complete=$complete/$EXPECTED stable=$stable/$STABLE_POLLS" \
        >>"$WATCH_LOG"
    if ((complete == EXPECTED)); then
        stable=$((stable + 1))
        if ((stable >= STABLE_POLLS)); then
            break
        fi
    else
        stable=0
    fi
    sleep "$INTERVAL"
done

echo "[$(date '+%F %T')] training matrix complete; starting postprocess" >>"$WATCH_LOG"
bash "$WORKSPACE/scripts/trainers/orchestration/postprocess_completed.sh" --run >>"$WATCH_LOG" 2>&1
echo "[$(date '+%F %T')] postprocess complete" >>"$WATCH_LOG"
