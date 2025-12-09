set -x

PROJECT_NAME="amo_grpo_math-500"

# EXPERIMENT_NAME="qwen3-4b_grpo"
EXPERIMENT_NAME="qwen3-4b_vanilla"

LATEST_STEP=$(cat checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/latest_checkpointed_iteration.txt)

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${LATEST_STEP}/actor \
    --target_dir checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${LATEST_STEP}/actor/huggingface
