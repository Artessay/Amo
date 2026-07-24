set -x

WORKSPACE=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
echo "Using workspace: $WORKSPACE"

DATASETS=(
    #"MATH-500" 
    "MATH-LightEval"
)

EXPERIMENTS=(
    # "qwen2.5-1.5b"
    # "qwen2.5-1.5b_grpo"
    # "qwen2.5-1.5b_gdpo"
    # "qwen2.5-1.5b_hvpo"
    
    # "qwen2.5-3b"
    # "qwen2.5-3b_grpo"
    # "qwen2.5-3b_gdpo"
    "qwen2.5-3b_hvpo"

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

REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_math/math_accuracy.py','$WORKSPACE/recipe/amo_math/math_conciseness.py','$WORKSPACE/recipe/amo_math/math_format.py']"

# Evaluation
for DATASET in "${DATASETS[@]}"; do
    for EXPERIMENT_NAME in "${EXPERIMENTS[@]}"; do
        OUTPUT_PATH="$WORKSPACE/results/$DATASET/$EXPERIMENT_NAME.parquet"
        if [ ! -f "$OUTPUT_PATH" ]; then
            echo "Output file not found: $OUTPUT_PATH"
            continue
        fi

        python3 -m verl.trainer.amo_eval \
            data.path=$OUTPUT_PATH \
            custom_reward_function.path=$REWARD_FUNCTION_PATH 
    done
done
