#!/usr/bin/env bash
set -euo pipefail

die() {
    echo "trainer launcher: $*" >&2
    exit 2
}

usage() {
    cat <<'EOF'
Internal usage: launch.sh METHOD DATASET [MODEL] [EPOCH] [--dry-run] [Hydra overrides...]

Public entry points live in scripts/trainers/<method>/run_<dataset>.sh.
MODEL: 1.5b | 3b | llama3b (canonical model tags are also accepted).
EOF
}

if (($# < 2)); then
    usage >&2
    exit 2
fi

METHOD=$1
DATASET=$2
shift 2

[[ $METHOD =~ ^[a-z0-9_]+$ ]] || die "invalid method id '$METHOD'"
[[ $DATASET =~ ^[a-z0-9-]+$ ]] || die "invalid dataset id '$DATASET'"

COMMON_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
TRAINERS_DIR=$(dirname "$COMMON_DIR")
WORKSPACE=$(dirname "$(dirname "$TRAINERS_DIR")")

case "${1:-}" in
    1.5b|qwen2.5-1.5b)
        MODEL=qwen2.5-1.5b
        shift
        ;;
    3b|qwen2.5-3b)
        MODEL=qwen2.5-3b
        shift
        ;;
    llama3b|llama3.2-3b)
        MODEL=llama3.2-3b
        shift
        ;;
    ""|*=*|--*)
        MODEL=${DEFAULT_MODEL:-qwen2.5-1.5b}
        ;;
    *)
        die "unknown MODEL '$1' (use 1.5b, 3b, or llama3b)"
        ;;
esac

case "$MODEL" in
    1.5b) MODEL=qwen2.5-1.5b ;;
    3b) MODEL=qwen2.5-3b ;;
    llama3b) MODEL=llama3.2-3b ;;
    qwen2.5-1.5b|qwen2.5-3b|llama3.2-3b) ;;
    *) die "invalid DEFAULT_MODEL '$MODEL'" ;;
esac

EPOCH_OVERRIDE=""
if [[ ${1:-} =~ ^[0-9]+$ ]]; then
    EPOCH_OVERRIDE=$1
    shift
fi

DRY_RUN=${DRY_RUN:-0}
if [[ ${1:-} == --dry-run ]]; then
    DRY_RUN=1
    shift
fi

USER_OVERRIDES=("$@")
for override in "${USER_OVERRIDES[@]}"; do
    case "$override" in
        trainer.project_name=*|trainer.experiment_name=*|trainer.default_local_dir=*|trainer.resume_mode=*|trainer.resume_from_path=*|+trainer.project_name=*|+trainer.experiment_name=*|+trainer.default_local_dir=*|++trainer.project_name=*|++trainer.experiment_name=*|++trainer.default_local_dir=*)
            die "'$override' changes artifact identity; use EXPERIMENT_NAME, CHECKPOINT_DIR, or RESUME_MODE explicitly"
            ;;
    esac
done
PYTHON_BIN=${AMO_PY:-python3}
TRAINER_VARIANT=${TRAINER_VARIANT:-}

BASE_PROFILE=$COMMON_DIR/base.sh
MODEL_PROFILE=$COMMON_DIR/models/$MODEL.sh
DATASET_PROFILE=$COMMON_DIR/datasets/$DATASET.sh
METHOD_PROFILE=$TRAINERS_DIR/$METHOD/method.sh

[[ -f $MODEL_PROFILE ]] || die "missing model profile: $MODEL_PROFILE"
[[ -f $DATASET_PROFILE ]] || die "missing dataset profile: $DATASET_PROFILE"
[[ -f $METHOD_PROFILE ]] || die "missing method profile: $METHOD_PROFILE"

# shellcheck source=/dev/null
source "$BASE_PROFILE"
# shellcheck source=/dev/null
source "$MODEL_PROFILE"
# shellcheck source=/dev/null
source "$DATASET_PROFILE"
# shellcheck source=/dev/null
source "$METHOD_PROFILE"

configure_base
configure_model
configure_dataset
configure_method

[[ -n ${ADV_ESTIMATOR:-} ]] || die "$METHOD_PROFILE did not set ADV_ESTIMATOR"
[[ -n ${REWARD_MANAGER:-} ]] || die "$METHOD_PROFILE did not set REWARD_MANAGER"
[[ -n ${MODEL_PATH:-} ]] || die "$MODEL_PROFILE did not set MODEL_PATH"
[[ -n ${MODEL_TAG:-} ]] || die "$MODEL_PROFILE did not set MODEL_TAG"
[[ -n ${PROJECT_NAME:-} ]] || die "$DATASET_PROFILE did not set PROJECT_NAME"
[[ -n ${RESULTS_DATASET:-} ]] || die "$DATASET_PROFILE did not set RESULTS_DATASET"
[[ -n ${TRAIN_FILES:-} ]] || die "$DATASET_PROFILE did not set TRAIN_FILES"
[[ -n ${VAL_FILES:-} ]] || die "$DATASET_PROFILE did not set VAL_FILES"
[[ -n ${REWARD_FUNCTION_PATH:-} ]] || die "$DATASET_PROFILE did not set REWARD_FUNCTION_PATH"

TOTAL_EPOCHS=${EPOCH_OVERRIDE:-$PROFILE_EPOCHS}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$PROFILE_TRAIN_BATCH_SIZE}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-$PROFILE_MAX_PROMPT_LENGTH}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$PROFILE_MAX_RESPONSE_LENGTH}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$PROFILE_PPO_MINI_BATCH_SIZE}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-$PROFILE_MICRO_BATCH_SIZE_PER_GPU}
NUM_NODES=${NUM_NODES:-$PROFILE_NUM_NODES}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-$PROFILE_NUM_GPUS_PER_NODE}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-$PROFILE_TENSOR_MODEL_PARALLEL_SIZE}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-$PROFILE_GPU_MEMORY_UTILIZATION}
ROLLOUT_N=${ROLLOUT_N:-$PROFILE_ROLLOUT_N}
SAVE_FREQ=${SAVE_FREQ:-$PROFILE_SAVE_FREQ}
TEST_FREQ=${TEST_FREQ:-$PROFILE_TEST_FREQ}
RESUME_MODE=${RESUME_MODE:-$PROFILE_RESUME_MODE}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-$PROFILE_VAL_BEFORE_TRAIN}
TRAINER_LOGGER=${TRAINER_LOGGER:-$PROFILE_LOGGER}

if [[ -n ${TRAIN_GPUS:-} ]]; then
    export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS
elif [[ -n $PROFILE_TRAIN_GPUS && -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
    export CUDA_VISIBLE_DEVICES=$PROFILE_TRAIN_GPUS
fi

METHOD_TAG=$METHOD
if [[ -n $TRAINER_VARIANT ]]; then
    METHOD_TAG=${METHOD}_${TRAINER_VARIANT}
fi
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${MODEL_TAG}_${METHOD_TAG}}
[[ $EXPERIMENT_NAME =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid experiment name '$EXPERIMENT_NAME'"
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$WORKSPACE/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME}
RESULTS_DIR=$WORKSPACE/results/$RESULTS_DATASET

check_path() {
    local required_path=$1
    if [[ -e $required_path ]]; then
        return 0
    fi
    if [[ $DRY_RUN == 1 ]]; then
        echo "[dry-run] warning: required path does not exist: $required_path" >&2
        return 0
    fi
    die "required path does not exist: $required_path"
}

check_path "$TRAIN_FILES"
check_path "$VAL_FILES"
check_path "$MODEL_PATH"

COMMAND=(
    "$PYTHON_BIN" -m verl.trainer.main_ppo
    "algorithm.adv_estimator=$ADV_ESTIMATOR"
    "amo_strategy.enable=True"
    "data.train_files=$TRAIN_FILES"
    "data.val_files=$VAL_FILES"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "+data.apply_chat_template_kwargs.enable_thinking=False"
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "actor_rollout_ref.actor.optim.lr=1e-5"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU"
    "actor_rollout_ref.actor.use_kl_loss=True"
    "actor_rollout_ref.actor.kl_loss_coef=0.001"
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
    "actor_rollout_ref.actor.entropy_coeff=0"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.fsdp_config.param_offload=False"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL"
    "actor_rollout_ref.rollout.mode=sync"
    "actor_rollout_ref.rollout.n=$ROLLOUT_N"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU"
    "actor_rollout_ref.ref.fsdp_config.param_offload=True"
    "algorithm.use_kl_in_reward=False"
    "reward_model.reward_manager=$REWARD_MANAGER"
    "custom_reward_function.path=$REWARD_FUNCTION_PATH"
    "trainer.critic_warmup=0"
    "trainer.logger=$TRAINER_LOGGER"
    "trainer.project_name=$PROJECT_NAME"
    "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.default_local_dir=$CHECKPOINT_DIR"
    "trainer.n_gpus_per_node=$NUM_GPUS_PER_NODE"
    "trainer.nnodes=$NUM_NODES"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.total_epochs=$TOTAL_EPOCHS"
    "trainer.resume_mode=$RESUME_MODE"
    "trainer.val_before_train=$VAL_BEFORE_TRAIN"
)

COMMAND+=("${MODEL_OVERRIDES[@]}")
COMMAND+=("${DATASET_OVERRIDES[@]}")
COMMAND+=("${METHOD_OVERRIDES[@]}")
COMMAND+=("${VARIANT_OVERRIDES[@]}")
COMMAND+=("${USER_OVERRIDES[@]}")

echo "Using workspace: $WORKSPACE"
echo "Experiment: $PROJECT_NAME/$EXPERIMENT_NAME"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Results: $RESULTS_DIR/${EXPERIMENT_NAME}.{parquet,json} (written by inference/evaluation)"

if [[ $DRY_RUN == 1 ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

cd "$WORKSPACE"
if [[ ${TRAINER_TRACE:-1} == 1 ]]; then
    set -x
fi
exec "${COMMAND[@]}"
