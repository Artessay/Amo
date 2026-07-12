#!/bin/bash
# [Amo] Smoke-test EVERY multi-objective baseline with a tiny 2-step run on the
# MATH task, to verify the full training pipeline (reward manager -> per-objective
# scores -> advantage estimator -> actor update) works end-to-end for each method.
#
# This is a *correctness* smoke test, NOT a comparison run: batch/steps are tiny.
# Each method logs to /tmp/amo_baseline_smoke/<method>.log and the script prints a
# PASS/FAIL summary at the end.
#
# Usage:
#   bash smoke_all_baselines.sh [MODEL]
# Env:
#   TRAIN_GPUS=0,1   GPUs to use (default 0,1)
set -u

MODEL=${1:-1.5b}
export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-0,1}

HERE=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
LOGDIR=/tmp/amo_baseline_smoke
mkdir -p "$LOGDIR"

METHODS=(
  ls tchebycheff
  gdpo_weighted rvpo
  mgda gapo
  lagrangian fair_stable ctwa dynamic_hv
  nsga2 smsemoa
)

# Tiny overrides shared by every smoke run.
SMOKE_OVERRIDES=(
  trainer.total_training_steps=2
  data.train_batch_size=16
  actor_rollout_ref.actor.ppo_mini_batch_size=16
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
  trainer.save_freq=-1
  trainer.test_freq=-1
  trainer.val_before_train=False
)

declare -A RESULT
for m in "${METHODS[@]}"; do
  echo "=========================================================="
  echo "[smoke] $m"
  echo "=========================================================="
  if bash "$HERE/run_baseline_math.sh" "$m" "$MODEL" 1 "${SMOKE_OVERRIDES[@]}" \
        > "$LOGDIR/$m.log" 2>&1; then
    RESULT[$m]="PASS"
  else
    RESULT[$m]="FAIL (see $LOGDIR/$m.log)"
  fi
  echo "[smoke] $m -> ${RESULT[$m]}"
done

echo ""
echo "==================== SMOKE SUMMARY ======================="
for m in "${METHODS[@]}"; do
  printf "  %-16s %s\n" "$m" "${RESULT[$m]}"
done
