set -x

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

if [ -n "$MODEL_PATH" ]; then
    unset MODEL_PATH
fi

# MODEL_PATH="/data/Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_PATH="/data/Qwen/Qwen2.5-3B-Instruct"
# MODEL_PATH="/data/meta-llama/Llama-3.2-3B-Instruct"

MODEL_PATH="/data/Qwen/Qwen3-4B"

EXPERIMENTS=(
    # "qwen2.5-1.5b"
    # "qwen2.5-1.5b_grpo"
    # "qwen2.5-1.5b_gdpo"
    # "qwen2.5-1.5b_hvpo"

    # "qwen2.5-3b"
    # "qwen2.5-3b_grpo"
    "qwen2.5-3b_tchebycheff"
    # "qwen2.5-3b_gdpo"
    "qwen2.5-3b_rvpo"
    # "qwen2.5-3b_hvpo"

    # "llama3.2-3b"
    # "llama3.2-3b_grpo"
    # "llama3.2-3b_gdpo"
    # "llama3.2-3b_hvpo"

    # "qwen2.5-1.5b_hvpo_distance"
    # "qwen2.5-1.5b_hvpo_euclidean"

    # "qwen2.5-1.5b_hvpo_lag1"
    # "qwen2.5-1.5b_hvpo_lag3"
    # "qwen2.5-1.5b_hvpo_lag7"
)

# Evaluation
for DATASET in "${DATASETS[@]}"; do
    for EXPERIMENT_NAME in "${EXPERIMENTS[@]}"; do
        DATASET_NAME="${DATASET,,}"
        DATA_PATH="$WORKSPACE/data/$DATASET/test.parquet"

        if [[ $EXPERIMENT_NAME == *_* ]]; then
            PROJECT_NAME="${PROJECT_PREFIX}_${DATASET_NAME}"
            STEP_FILE_PATH="${WORKSPACE}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/latest_checkpointed_iteration.txt"
            
            if [ ! -f "$STEP_FILE_PATH" ]; then
                echo "Step file not found: $STEP_FILE_PATH"
                continue
            fi

            LATEST_STEP=$(cat "$STEP_FILE_PATH")
            MODEL_PATH="${WORKSPACE}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${LATEST_STEP}/actor/merge"

            if [ ! -d "$MODEL_PATH" ]; then
                echo "Model path not found: $MODEL_PATH"
                continue
            fi
        else
            if [ ! -d "$MODEL_PATH" ]; then
                echo "Error: MODEL_PATH not found for $EXPERIMENT_NAME"
                continue
            else
                echo "Using predefined MODEL_PATH: $MODEL_PATH"
            fi
        fi

        mkdir -p "$WORKSPACE/results/$DATASET"
        OUTPUT_PATH="$WORKSPACE/results/$DATASET/$EXPERIMENT_NAME.parquet"
        # if [ -f "$OUTPUT_PATH" ]; then
        #     echo "Output file already exists: $OUTPUT_PATH"
        #     continue
        # fi
        
        python3 playground/generation.py \
            --model $MODEL_PATH \
            --data $DATA_PATH \
            --output $OUTPUT_PATH
    done
done
