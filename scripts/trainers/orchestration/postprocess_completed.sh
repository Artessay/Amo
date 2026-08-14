#!/usr/bin/env bash
# Discover successfully trained priority-baseline cells and postprocess the
# ones that do not yet have canonical result JSON files.
#
# Usage:
#   bash postprocess_completed.sh --list
#   POSTPROCESS_GPUS=4,5 bash postprocess_completed.sh --run
set -euo pipefail

MODE=${1:---list}
case "$MODE" in --list|--run) ;; *) echo "usage: $0 [--list|--run]" >&2; exit 2 ;; esac

WORKSPACE=$(dirname "$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")")
CELL_RUNNER=$WORKSPACE/scripts/trainers/tools/postprocess_cell.sh
LOGDIR=$WORKSPACE/train_logs/postprocess
LEDGER=$LOGDIR/progress.log
mkdir -p "$LOGDIR"

declare -A SEEN=()
CELLS=()
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
    [[ -n $dataset && -n $experiment ]] || {
        echo "warning: malformed completion marker: $marker" >&2
        continue
    }
    key=$dataset/$experiment
    [[ -z ${SEEN[$key]:-} ]] || continue
    SEEN[$key]=1
    CELLS+=("$key")
done < <(find "$WORKSPACE/train_logs/priority_baselines" -type f \
    -path '*/completed_train/*.done' -print | sort)

result_json() {
    local dataset=$1 experiment=$2 result_dir
    case "$dataset" in
        math-lighteval) result_dir=MATH-LightEval ;;
        news) result_dir=CNN_DailyMail ;;
        pku-saferlhf) result_dir=PKU-SafeRLHF ;;
        rlla) result_dir=RLLA ;;
        *) return 1 ;;
    esac
    echo "$WORKSPACE/results/$result_dir/$experiment.json"
}

pending=0
for key in "${CELLS[@]}"; do
    dataset=${key%%/*}
    experiment=${key#*/}
    json=$(result_json "$dataset" "$experiment")
    if [[ -s $json ]]; then
        printf 'DONE\t%s\t%s\t%s\n' "$dataset" "$experiment" "$json"
        continue
    fi
    pending=$((pending + 1))
    printf 'PENDING\t%s\t%s\t%s\n' "$dataset" "$experiment" "$json"
    [[ $MODE == --run ]] || continue

    [[ -n ${POSTPROCESS_GPUS:-} ]] || {
        echo "POSTPROCESS_GPUS is required with --run" >&2
        exit 2
    }
    cell_log=$LOGDIR/${experiment}.${dataset}.log
    printf '[%s] START dataset=%s experiment=%s log=%s\n' \
        "$(date '+%F %T')" "$dataset" "$experiment" "$cell_log" | tee -a "$LEDGER"
    if bash "$CELL_RUNNER" "$dataset" "$experiment" >>"$cell_log" 2>&1; then
        printf '[%s] OK dataset=%s experiment=%s\n' \
            "$(date '+%F %T')" "$dataset" "$experiment" | tee -a "$LEDGER"
    else
        rc=$?
        printf '[%s] FAIL rc=%s dataset=%s experiment=%s log=%s\n' \
            "$(date '+%F %T')" "$rc" "$dataset" "$experiment" "$cell_log" | tee -a "$LEDGER"
        exit "$rc"
    fi
done

printf 'discovered=%d pending=%d mode=%s\n' "${#CELLS[@]}" "$pending" "$MODE"
