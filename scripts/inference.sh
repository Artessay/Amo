set -x

WORKSPACE=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
echo "Using workspace: $WORKSPACE"

PROJECT_PREFIX="amo_grpo"

DATASETS=(
    # "MATH-500"
    # "MATH-lighteval"
    "PKU-SafeRLHF"
    # "cnn_dailymail"
)

# MODEL_PATH="/data/Qwen/Qwen2.5-1.5B-Instruct"

EXPERIMENTS=(
    # "qwen2.5-1.5b"
    # "qwen2.5-1.5b_grpo"
    # "qwen2.5-1.5b_vanilla"
    "qwen2.5-1.5b_hv"

    # "qwen2.5-3b"
    
    # "llama3-3b"
)

# Evaluation
for DATASET in "${DATASETS[@]}"; do
    for EXPERIMENT_NAME in "${EXPERIMENTS[@]}"; do
        DATASET_NAME="${DATASET,,}"
        DATA_PATH="$WORKSPACE/data/$DATASET/test.parquet"
        PROJECT_NAME="${PROJECT_PREFIX}_${DATASET_NAME}"
        STEP_FILE_PATH="${WORKSPACE}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/latest_checkpointed_iteration.txt"

        if [ ! -f $STEP_FILE_PATH ]; then
            echo "Step file not found: $STEP_FILE_PATH"
            continue
        fi
        LATEST_STEP=$(cat $STEP_FILE_PATH)
        MODEL_PATH="${WORKSPACE}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${LATEST_STEP}/actor/merge"

        OUTPUT_PATH="$WORKSPACE/results/$DATASET/$EXPERIMENT_NAME.parquet"
        
        python3 playground/generation.py \
            --model $MODEL_PATH \
            --data $DATA_PATH \
            --output $OUTPUT_PATH
    done
done