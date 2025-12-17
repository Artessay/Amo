set -x

WORKSPACE=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
echo "Using workspace: $WORKSPACE"

PROJECT_PREFIX="amo_grpo"

DATASETS=(
    "MATH-500" 
    "MATH-lighteval"
    "PKU-SafeRLHF"
)

# BASE_MODEL="/data/Qwen/Qwen2.5-1.5B-Instruct"
# EXPERIMENTS=(
#     "qwen2.5-1.5b_vanilla"
# )

BASE_MODEL="/data/Qwen/Qwen3-4B"
EXPERIMENTS=(
    "qwen3-4b_grpo"
    "qwen3-4b_vanilla"
)


# Merge LoRA checkpoints
for DATASET in "${DATASETS[@]}"; do
    for EXPERIMENT_NAME in "${EXPERIMENTS[@]}"; do
        DATASET_NAME="${DATASET,,}"
        PROJECT_NAME="${PROJECT_PREFIX}_${DATASET_NAME}"
        STEP_FILE_PATH="${WORKSPACE}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/latest_checkpointed_iteration.txt"

        if [ ! -f $STEP_FILE_PATH ]; then
            echo "Step file not found: $STEP_FILE_PATH"
            continue
        fi

        LATEST_STEP=$(cat $STEP_FILE_PATH)

        python3 playground/lora_merger.py \
            --model_path $BASE_MODEL \
            --adapter_path ${WORKSPACE}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${LATEST_STEP}/actor/lora_adapter \
            --save_path ${WORKSPACE}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${LATEST_STEP}/actor/merge

    done
done
