set -x

WORKSPACE=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
echo "Using workspace: $WORKSPACE"

PROJECT_PREFIX="amo_grpo"

DATASETS=(
    "MATH-500" 
    # "MATH-lighteval"
)

EXPERIMENTS=(
    # "qwen3-4b"
    # "qwen3-4b_grpo"
    # "qwen3-4b_vanilla"
    # "llama3-3b"
    # "qwen2.5-1.5b"
    "qwen2.5-1.5b_vanilla"
    # "qwen2.5-3b"
)

REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_math/math_accuracy.py','$WORKSPACE/recipe/amo_math/math_conciseness.py','$WORKSPACE/recipe/amo_math/math_format.py']"

# Evaluation
for DATASET in "${DATASETS[@]}"; do
    for EXPERIMENT_NAME in "${EXPERIMENTS[@]}"; do
        DATASET_NAME="${DATASET,,}"
        DATA_PATH="$WORKSPACE/data/$DATASET/test.parquet"
        MODEL_PATH="$WORKSPACE/checkpoints/${PROJECT_PREFIX}_${DATASET_NAME}/${EXPERIMENT_NAME}/merge"
        OUTPUT_PATH="$WORKSPACE/results/$DATASET/$EXPERIMENT_NAME.parquet"
        
        # python3 -m verl.trainer.amo_generation \
        python3 verl/trainer/amo_generation.py \
            --model $MODEL_PATH \
            --data $DATA_PATH \
            --output $OUTPUT_PATH \
            --max_tokens 2048

    done
done