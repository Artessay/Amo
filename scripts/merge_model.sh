# Merge LoRA checkpoints for All datasets

# set -x

WORKSPACE=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
echo "Using workspace: $WORKSPACE"

PROJECT_PREFIX="amo"

DATASETS=(
    # "MATH-500"
    "MATH-LightEval"
    # "PKU-SafeRLHF"
    # "RLLA"
    # "CNN_DailyMail"
)

#BASE_MODEL="/data/Qwen/Qwen2.5-1.5B-Instruct"
#EXPERIMENTS=(
    # "qwen2.5-1.5b_grpo"
    # "qwen2.5-1.5b_gdpo"
    # "qwen2.5-1.5b_hvpo"
    # "qwen2.5-1.5b_hvpo_distance"
    # "qwen2.5-1.5b_hvpo_euclidean"
    # "qwen2.5-1.5b_hvpo_lag1"
    # "qwen2.5-1.5b_hvpo_lag3"
    # "qwen2.5-1.5b_hvpo_lag7"
#)

BASE_MODEL="/data/Qwen/Qwen2.5-3B-Instruct"
EXPERIMENTS=(
    # "qwen2.5-3b_grpo"
    "qwen2.5-3b_tchebycheff"
    # "qwen2.5-3b_gdpo"
    "qwen2.5-3b_rvpo"
    # "qwen2.5-3b_hvpo"
)

#  BASE_MODEL="/data/meta-llama/Llama-3.2-3B-Instruct"
#  EXPERIMENTS=(
#     # "llama3.2-3b_grpo"
#     # "llama3.2-3b_gdpo"
#     # "llama3.2-3b_hvpo"
#  )


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
        ADAPTER_PATH="${WORKSPACE}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${LATEST_STEP}/actor/lora_adapter"
        SAVE_PATH="${WORKSPACE}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${LATEST_STEP}/actor/merge"

        if [ -d $SAVE_PATH ]; then
            echo "Save path already exists: $SAVE_PATH"
            continue
        fi

        if [ ! -d $ADAPTER_PATH ]; then
            echo "Merging FSDP checkpoints for $PROJECT_NAME/$EXPERIMENT_NAME"
            ACTOR_PATH="${WORKSPACE}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${LATEST_STEP}/actor/"
            python playground/legacy_model_merger.py merge \
                --backend fsdp \
                --local_dir $ACTOR_PATH \
                --target_dir $SAVE_PATH
        else
            echo "Merging LoRA checkpoints for $PROJECT_NAME/$EXPERIMENT_NAME"
            python3 playground/lora_merger.py \
                --model_path $BASE_MODEL \
                --adapter_path $ADAPTER_PATH \
                --save_path $SAVE_PATH
        fi
    done
done
