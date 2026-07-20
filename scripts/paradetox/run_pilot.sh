#!/usr/bin/env bash

set -euo pipefail

if (($# < 1)); then
    echo "Usage: $0 {grpo|hvpo} [Hydra overrides ...]" >&2
    exit 2
fi

METHOD=$1
shift
case "$METHOD" in
    grpo)
        ADV_ESTIMATOR=grpo
        REWARD_MANAGER=amo_vanilla
        ;;
    hvpo)
        ADV_ESTIMATOR=hvpo
        REWARD_MANAGER=amo_hvpo
        ;;
    *)
        echo "Unknown method: $METHOD (expected grpo or hvpo)" >&2
        exit 2
        ;;
esac

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")
MODEL_PATH=${MODEL_PATH:-/data/Qwen/Qwen2.5-1.5B-Instruct}
TRAIN_FILES=${TRAIN_FILES:-$WORKSPACE/data/ParaDetox/train.parquet}
VAL_FILES=${VAL_FILES:-$WORKSPACE/data/ParaDetox/test.parquet}
SEED=${SEED:-42}
# Keep prompt order and the held-out subset fixed across model seeds. SEED is
# reserved for actor/rollout randomness; override DATA_SEED only deliberately.
DATA_SEED=${DATA_SEED:-42}
STEPS=${STEPS:-10}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
ROLLOUT_N=${ROLLOUT_N:-8}
VAL_N=${VAL_N:-4}
VAL_SAMPLES=${VAL_SAMPLES:-64}
RUN_TAG=${RUN_TAG:-pilot10}
RESULT_ROOT=${RESULT_ROOT:-$WORKSPACE/results/ParaDetox/$RUN_TAG}
EXPERIMENT_NAME=qwen2.5-1.5b_${METHOD}_seed${SEED}
RUN_DIR=$RESULT_ROOT/$EXPERIMENT_NAME

for required in "$MODEL_PATH/config.json" "$TRAIN_FILES" "$VAL_FILES"; do
    if [[ ! -f "$required" ]]; then
        echo "Required file not found: $required" >&2
        exit 1
    fi
done

if [[ -e "$RUN_DIR" ]]; then
    echo "Run directory already exists: $RUN_DIR" >&2
    echo "Choose a new RUN_TAG to avoid mixing old and new validation steps." >&2
    exit 1
fi

REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_detox/detox_sta.py','$WORKSPACE/recipe/amo_detox/detox_sim.py','$WORKSPACE/recipe/amo_detox/detox_fluency.py']"

export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export PYTHONPATH="$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
export STA_TARGET_PORT=${STA_TARGET_PORT:-${DETOX_STA_PORT:-50060}}
export SIM_TARGET_PORT=${SIM_TARGET_PORT:-${DETOX_SIM_PORT:-50061}}
export FL_TARGET_PORT=${FL_TARGET_PORT:-${DETOX_FL_PORT:-50062}}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="$ADV_ESTIMATOR" \
    algorithm.use_kl_in_reward=False \
    amo_strategy.enable=True \
    amo_strategy.hv_config.pareto_front_scope=intra_group \
    amo_strategy.hv_config.normalize_objectives=True \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_max_samples="$VAL_SAMPLES" \
    data.seed="$DATA_SEED" \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.seed="$SEED" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    +actor_rollout_ref.rollout.seed="$SEED" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
    actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager="$REWARD_MANAGER" \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=amo_paradetox_pilot \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$STEPS" \
    trainer.val_before_train=True \
    trainer.test_freq="$STEPS" \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.default_local_dir="$RUN_DIR/checkpoints" \
    trainer.validation_data_dir="$RUN_DIR/validation" \
    trainer.rollout_data_dir="$RUN_DIR/rollouts" \
    "$@"
