#!/bin/bash
# [Amo] Unified launcher for the multi-objective *baseline* methods on the MATH
# task (3 objectives: accuracy, conciseness, format — all local, no reward
# server). Every baseline reuses the SAME model / data / rollout n / KL / token
# budget as the GRPO/GDPO/HVPO scripts; only the multi-objective credit differs,
# so results are directly comparable in one controlled study.
#
# Usage:
#   bash run_baseline_math.sh <METHOD> [MODEL] [EPOCH] [extra hydra overrides...]
#
#   METHOD (required):
#     ls              linear scalarization (weighted sum / MORLHF), GRPO adv
#     tchebycheff     (augmented) Tchebycheff scalarization, GRPO adv
#     gdpo_weighted   weighted GDPO (per-objective weighted group z-score)
#     rvpo            reward-variance (soft-min over objectives) advantage
#     mgda            min-norm advantage aggregation (advantage-space proxy)
#     gapo            grad-norm rescaled MGDA (advantage-space proxy)
#     lagrangian      Safe-RLHF-style constrained (dual ascent), GRPO adv
#     fair_stable     Fair-and-Stable mirror-descent reward composition, GRPO adv
#     ctwa            covariance-targeted weight adaptation, GRPO adv
#     dynamic_hv      HV-guided group-level dynamic reward weighting, GRPO adv
#     nsga2           NSGA-II-style (rank + crowding) response credit, GRPO adv
#     smsemoa         SMS-EMOA-style exclusive-HV response credit, GRPO adv
#
#   MODEL : 1.5b (default) | 3b
#   EPOCH : total_epochs (default 50). Pass a small trainer.total_training_steps
#           override at the end for a smoke test, e.g. `... trainer.total_training_steps=2`.
set -x
set -e

METHOD=${1:?need METHOD (see header)}
MODEL=${2:-1.5b}
EPOCH=${3:-50}
shift $(( $# < 3 ? $# : 3 ))

# amo env python (base env lacks verl deps). Override with AMO_PY if needed.
PY=${AMO_PY:-/home/rihongqiu/data/miniconda3/envs/amo/bin/python}

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")
echo "Using workspace: $WORKSPACE"

case "$MODEL" in
  1.5b) MODEL_PATH="/data/Qwen/Qwen2.5-1.5B-Instruct"; MODEL_TAG="qwen2.5-1.5b" ;;
  3b)   MODEL_PATH="/data/Qwen/Qwen2.5-3B-Instruct";   MODEL_TAG="qwen2.5-3b"  ;;
  *) echo "bad MODEL $MODEL"; exit 1 ;;
esac

PROJECT_NAME="amo_math-lighteval"
TRAIN_FILES="$WORKSPACE/data/MATH-lighteval/train.parquet"
VAL_FILES="$WORKSPACE/data/MATH-lighteval/val.parquet"
REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_math/math_accuracy.py','$WORKSPACE/recipe/amo_math/math_conciseness.py','$WORKSPACE/recipe/amo_math/math_format.py']"

# --- Map METHOD -> (adv_estimator, reward_manager, method-specific overrides) ---
# The three MATH objectives are already in [0,1], so default reference/ideal
# points (origin / all-ones) are correct.
EXTRA=()
case "$METHOD" in
  ls)
    ADV="grpo"; RM="amo_scalarize"
    EXTRA=( amo_strategy.scalarize_config.method=linear )
    ;;
  tchebycheff)
    ADV="grpo"; RM="amo_scalarize"
    EXTRA=( amo_strategy.scalarize_config.method=tchebycheff
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
    ADV="grpo"; RM="amo_adaptive"
    EXTRA=( amo_strategy.adaptive_config.method=lagrangian
            amo_strategy.adaptive_config.primary_index=0
            "amo_strategy.adaptive_config.budgets=[0.0,0.5,0.5]"
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
            "amo_strategy.adaptive_config.cov_targets=[0.0,0.0,0.0]"
            amo_strategy.adaptive_config.weight_lr=0.1 )
    ;;
  dynamic_hv)
    ADV="grpo"; RM="amo_adaptive"
    EXTRA=( amo_strategy.adaptive_config.method=dynamic_hv )
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
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-32}
TENSOR_MODEL_PARALLEL_SIZE=1

$PY -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV \
    amo_strategy.enable=True \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
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
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=$EPOCH "$@"
