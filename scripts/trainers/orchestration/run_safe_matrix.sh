#!/bin/bash
# [Amo] Sequential driver for the full PKU-SafeRLHF trainer matrix:
#   { qwen2.5-1.5b, qwen2.5-3b, llama3.2-3b } x { 15 methods }
# For each cell: train (scripts/trainers/<method>/run_pku-saferlhf.sh), then
# evaluate (scripts/trainers/tools/eval_safe.sh) and write
# results/PKU-SafeRLHF/<MODEL_TAG>_<METHOD>.json.
#
# Only GPU 0,1 are used (shared with the reward servers), so runs are strictly
# serial. The driver is idempotent / resumable:
#   * a cell whose result JSON already exists is skipped;
#   * training uses the method entry point's resume_mode=auto behavior;
#   * a failing cell is logged and skipped; the matrix continues.
#
# Per-cell logs: train_logs/safe_baselines/<exp>.{train,eval}.log
# Overall progress ledger: train_logs/safe_baselines/matrix_progress.log
#
# Usage:
#   bash scripts/trainers/orchestration/run_safe_matrix.sh [MODELS] [METHODS] [EPOCH]
#     MODELS  : space/comma list, default "1.5b 3b llama3b"
#     METHODS : space/comma list, default GRPO/GDPO/HVPO + all 12 baselines
#     EPOCH   : total_epochs per run, default 1
set -u

WORKSPACE=$(dirname "$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")")
cd "$WORKSPACE"
export AMO_PY=${AMO_PY:-/home/rihongqiu/data/miniconda3/envs/amo/bin/python}
LOGDIR=$WORKSPACE/train_logs/safe_baselines
mkdir -p "$LOGDIR"
LEDGER=$LOGDIR/matrix_progress.log

MODELS_IN=${1:-"1.5b 3b llama3b"}
METHODS_IN=${2:-"grpo gdpo hvpo tchebycheff rvpo mgda gapo lagrangian fair_stable ctwa dynamic_hv nsga2 smsemoa"}
EPOCH=${3:-1}
MODELS=$(echo "$MODELS_IN" | tr ',' ' ')
METHODS=$(echo "$METHODS_IN" | tr ',' ' ')

model_tag() {
  case "$1" in
    1.5b)    echo "qwen2.5-1.5b" ;;
    3b)      echo "qwen2.5-3b" ;;
    llama3b) echo "llama3.2-3b" ;;
    *) echo "UNKNOWN" ;;
  esac
}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LEDGER"; }

log "=== safe trainer matrix START ==="
log "models: $MODELS"
log "methods: $METHODS"
log "epoch: $EPOCH"

TOTAL=0; DONE=0; SKIP=0; FAIL=0
for MODEL in $MODELS; do
  TAG=$(model_tag "$MODEL")
  for METHOD in $METHODS; do
    TOTAL=$((TOTAL+1))
    EXP="${TAG}_${METHOD}"
    RESULT_JSON=$WORKSPACE/results/PKU-SafeRLHF/${EXP}.json
    TRAIN_LOG=$LOGDIR/${EXP}.train.log
    EVAL_LOG=$LOGDIR/${EXP}.eval.log

    if [ -f "$RESULT_JSON" ]; then
      log "SKIP  $EXP (result exists: $RESULT_JSON)"
      SKIP=$((SKIP+1)); continue
    fi

    case "$METHOD" in
      gdpo|hvpo|grpo|tchebycheff|rvpo|mgda|gapo|lagrangian|fair_stable|ctwa|dynamic_hv|nsga2|smsemoa) ;;
      *) log "TRAIN FAIL $EXP (unsupported method: $METHOD)"; FAIL=$((FAIL+1)); continue ;;
    esac
    METHOD_SCRIPT=$WORKSPACE/scripts/trainers/$METHOD/run_pku-saferlhf.sh
    if [ ! -f "$METHOD_SCRIPT" ]; then
      log "TRAIN FAIL $EXP (missing method entry point: $METHOD_SCRIPT)"
      FAIL=$((FAIL+1)); continue
    fi

    log "TRAIN $EXP -> $TRAIN_LOG"
    if bash "$METHOD_SCRIPT" "$MODEL" "$EPOCH" > "$TRAIN_LOG" 2>&1; then
      log "TRAIN ok   $EXP"
    else
      log "TRAIN FAIL $EXP (see $TRAIN_LOG) -- skipping eval"
      FAIL=$((FAIL+1)); continue
    fi

    log "EVAL  $EXP -> $EVAL_LOG"
    if bash "$WORKSPACE/scripts/trainers/tools/eval_safe.sh" "$EXP" "$MODEL" \
         > "$EVAL_LOG" 2>&1; then
      log "EVAL ok    $EXP"
      DONE=$((DONE+1))
    else
      log "EVAL FAIL  $EXP (see $EVAL_LOG)"
      FAIL=$((FAIL+1))
    fi

    # Refresh the aggregate table after every completed cell.
    "${AMO_PY:-/home/rihongqiu/data/miniconda3/envs/amo/bin/python}" \
      "$WORKSPACE/scripts/trainers/tools/aggregate_safe.py" >> "$LEDGER" 2>&1 || true
  done
done

log "=== safe trainer matrix END: total=$TOTAL done=$DONE skip=$SKIP fail=$FAIL ==="
