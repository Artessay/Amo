#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=$(dirname "$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")")
cd "$WORKSPACE"

PY=${AMO_PY:-/data2/home/qrh/miniconda3/envs/amo/bin/python}
LOGDIR=$WORKSPACE/train_logs/postprocess
LEDGER=$LOGDIR/news_then_resume.log
FIRST_JSON=$WORKSPACE/results/CNN_DailyMail/qwen2.5-1.5b_dynamic_hv.json
mkdir -p "$LOGDIR"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$LEDGER"
}

log "waiting for dynamic_hv News postprocess"
while [[ ! -s $FIRST_JSON ]]; do
    sleep 60
done

for experiment in qwen2.5-1.5b_grpo qwen2.5-1.5b_gdpo; do
    log "postprocess start: $experiment/news"
    POSTPROCESS_GPUS=4,5 \
    POSTPROCESS_SERVER_GPUS=4,5 \
    GPU_MEM_UTIL=0.8 \
    AMO_PY="$PY" \
        bash scripts/trainers/tools/postprocess_cell.sh news "$experiment" \
        >>"$LOGDIR/$experiment.news.log" 2>&1
    log "postprocess complete: $experiment/news"
done

log "resuming qwen2.5-1.5b training queue with dual-GPU News server"
export AMO_PY="$PY"
export BASELINE_MODEL=1.5b
export BASELINE_METHODS="dynamic_hv grpo gdpo rvpo tchebycheff ctwa lagrangian fair_stable mgda gapo nsga2 smsemoa"
export BASELINE_DATASETS="math-lighteval news"
export BASELINE_QUEUE_NAME=default
export TRAIN_GPUS=4,5
export SAFE_SERVER_GPUS=2,3
export NEWS_SERVER_GPUS=4,5
exec bash scripts/trainers/orchestration/run_priority_baselines.sh
