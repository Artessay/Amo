#!/bin/bash
# [Amo] Unified launcher for the multi-objective *baseline* methods on the
# PKU-SafeRLHF task (2 objectives: safe_helpfulness, safe_harmlessness -- served
# by the reward-model gRPC servers on GPU 0/1). Safe-task analog of
# run_baseline_math.sh: every baseline reuses the SAME model / data / rollout n /
# KL / token budget as the GRPO/GDPO/HVPO safe scripts; only the multi-objective
# credit differs, so results are directly comparable in one controlled study.
#
# The two Safe-RLHF reward objectives are *unbounded* reward-model logits
# (~[-4, +6]), so scale-sensitive baselines (ls / tchebycheff / lagrangian /
# dynamic_hv) are wired to the FROZEN calibration in
# results/PKU-SafeRLHF/safe_calibration.json (produced by calibrate_safe.py).
# Scale-invariant baselines (gdpo_weighted / rvpo / mgda / gapo / nsga2 /
# smsemoa / fair_stable / ctwa) consume the raw scores, exactly like GRPO/GDPO.
#
# HARD GPU CONSTRAINT: training runs ONLY on GPU 0,1 (shared with the reward
# servers). Never touches GPU 2,3.
#
# Usage:
#   bash run_baseline_safe.sh <METHOD> [MODEL] [EPOCH] [extra hydra overrides...]
#
#   METHOD (required): ls tchebycheff gdpo_weighted rvpo mgda gapo
#                      lagrangian fair_stable ctwa dynamic_hv nsga2 smsemoa
#   MODEL : 1.5b (default) | 3b | llama3b
#   EPOCH : total_epochs (default 1 == one full pass, ~144 steps @ batch 512).
#           For a smoke test append e.g. `trainer.total_training_steps=2`.
set -x
set -e

METHOD=${1:?need METHOD (see header)}
MODEL=${2:-1.5b}
EPOCH=${3:-1}
shift $(( $# < 3 ? $# : 3 ))

# amo env python (base env lacks verl deps). Override with AMO_PY if needed.
PY=${AMO_PY:-/home/rihongqiu/data/miniconda3/envs/amo/bin/python}

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")
echo "Using workspace: $WORKSPACE"

# --- HARD GPU CONSTRAINT: only GPU 0,1 (reward servers live here too) ---
export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-0,1}

case "$MODEL" in
  1.5b)    MODEL_PATH="/data/Qwen/Qwen2.5-1.5B-Instruct";       MODEL_TAG="qwen2.5-1.5b"; DEF_MICRO=16 ;;
  3b)      MODEL_PATH="/data/Qwen/Qwen2.5-3B-Instruct";         MODEL_TAG="qwen2.5-3b";   DEF_MICRO=16 ;;
  llama3b) MODEL_PATH="/data/meta-llama/Llama-3.2-3B-Instruct"; MODEL_TAG="llama3.2-3b";  DEF_MICRO=16 ;;
  *) echo "bad MODEL $MODEL (use 1.5b|3b|llama3b)"; exit 1 ;;
esac

PROJECT_NAME="amo_pku-saferlhf"
TRAIN_FILES="$WORKSPACE/data/PKU-SafeRLHF/train.parquet"
VAL_FILES="$WORKSPACE/data/PKU-SafeRLHF/test.parquet"
REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_safe/safe_helpfulness.py','$WORKSPACE/recipe/amo_safe/safe_harmlessness.py']"

# --- Load the frozen calibration constants (fair-comparison protocol) ---
CALIB_JSON="$WORKSPACE/results/PKU-SafeRLHF/safe_calibration.json"
if [ ! -f "$CALIB_JSON" ]; then
  echo "[safe] MISSING calibration $CALIB_JSON -- run calibrate_safe.py first"; exit 1
fi
read_calib() { $PY -c "import json;d=json.load(open('$CALIB_JSON'));print($1)"; }
CALIB_LOWER=$(read_calib "str(d['calib_lower']).replace(' ','')")   # [lo_help,lo_harm]
CALIB_UPPER=$(read_calib "str(d['calib_upper']).replace(' ','')")
CALIB_IDEAL=$(read_calib "str(d['ideal']).replace(' ','')")
HV_REF=$(read_calib "str(d['hv_reference']).replace(' ','')")
HARM_BUDGET=$(read_calib "d['harmless_budget']")
# Lagrangian budgets vector: [primary(unused)=0, harmless_budget]
LAG_BUDGETS="[0.0,${HARM_BUDGET}]"

# --- Map METHOD -> (adv_estimator, reward_manager, method-specific overrides) ---
# Scale-sensitive methods use the calibration; scale-invariant ones use raw scores.
EXTRA=()
case "$METHOD" in
  ls)
    ADV="grpo"; RM="amo_scalarize"
    EXTRA=( amo_strategy.scalarize_config.method=linear
            amo_strategy.scalarize_config.normalize=affine
            "amo_strategy.scalarize_config.calib_lower=${CALIB_LOWER}"
            "amo_strategy.scalarize_config.calib_upper=${CALIB_UPPER}" )
    ;;
  tchebycheff)
    ADV="grpo"; RM="amo_scalarize"
    EXTRA=( amo_strategy.scalarize_config.method=tchebycheff
            amo_strategy.scalarize_config.normalize=affine
            "amo_strategy.scalarize_config.calib_lower=${CALIB_LOWER}"
            "amo_strategy.scalarize_config.calib_upper=${CALIB_UPPER}"
            "amo_strategy.scalarize_config.ideal=${CALIB_IDEAL}"
            amo_strategy.scalarize_config.rho=0.05 )
    ;;
  gdpo_weighted)
    ADV="gdpo_weighted"; RM="amo_vanilla"
    ;;
  rvpo)
    ADV="rvpo"; RM="amo_vanilla"
    EXTRA=( algorithm.rvpo_k=1.0 )
    ;;
  mgda)
    ADV="mgda"; RM="amo_vanilla"
    ;;
  gapo)
    ADV="gapo"; RM="amo_vanilla"
    EXTRA=( algorithm.gapo_p=1.0 )
    ;;
  lagrangian)
    # Safe-RLHF: maximize helpfulness (index 0) s.t. harmlessness (index 1) >= budget.
    ADV="grpo"; RM="amo_adaptive"
    EXTRA=( amo_strategy.adaptive_config.method=lagrangian
            amo_strategy.adaptive_config.primary_index=0
            "amo_strategy.adaptive_config.budgets=${LAG_BUDGETS}"
            amo_strategy.adaptive_config.lambda_lr=0.05 )
    ;;
  fair_stable)
    ADV="grpo"; RM="amo_adaptive"
    EXTRA=( amo_strategy.adaptive_config.method=fair_stable
            amo_strategy.adaptive_config.weight_lr=0.1 )
    ;;
  ctwa)
    ADV="grpo"; RM="amo_adaptive"
    EXTRA=( amo_strategy.adaptive_config.method=ctwa
            "amo_strategy.adaptive_config.cov_targets=[0.0,0.0]"
            amo_strategy.adaptive_config.weight_lr=0.1 )
    ;;
  dynamic_hv)
    ADV="grpo"; RM="amo_adaptive"
    EXTRA=( amo_strategy.adaptive_config.method=dynamic_hv
            "amo_strategy.adaptive_config.hv_reference_point=${HV_REF}" )
    ;;
  nsga2)
    ADV="grpo"; RM="amo_pareto"
    EXTRA=( amo_strategy.pareto_config.method=nsga2 )
    ;;
  smsemoa)
    ADV="grpo"; RM="amo_pareto"
    EXTRA=( amo_strategy.pareto_config.method=smsemoa )
    ;;
  *) echo "bad METHOD $METHOD (see header)"; exit 1 ;;
esac

EXPERIMENT_NAME="${MODEL_TAG}_${METHOD}"

NUM_NODES=1
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-2}
# Conservative defaults so training coexists with the reward servers on GPU 0,1
# without OOM over a long unattended run. Override via env if you have headroom.
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-$DEF_MICRO}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}
TENSOR_MODEL_PARALLEL_SIZE=1

$PY -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV \
    amo_strategy.enable=True \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=$RM \
    custom_reward_function.path=$REWARD_FUNCTION_PATH \
    "${EXTRA[@]}" \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS_PER_NODE \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=${SAVE_FREQ:-50} \
    trainer.test_freq=${TEST_FREQ:-50} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-False} \
    trainer.total_epochs=$EPOCH \
    trainer.resume_mode=${RESUME_MODE:-auto} \
    "$@"
