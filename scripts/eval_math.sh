set -x

WORKSPACE=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
echo "Using workspace: $WORKSPACE"

PROJECT_PREFIX="amo_grpo"

DATASETS=(
    "MATH-500" 
    "MATH-lighteval"
)

EXPERIMENTS=(
    # "qwen2.5-1.5b"
    "qwen2.5-1.5b_grpo"
    # "qwen2.5-1.5b_vanilla"
    # "qwen2.5-3b"
    # "llama3-3b"
)

REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_math/math_accuracy.py','$WORKSPACE/recipe/amo_math/math_conciseness.py','$WORKSPACE/recipe/amo_math/math_format.py']"

# Evaluation
for DATASET in "${DATASETS[@]}"; do
    for EXPERIMENT_NAME in "${EXPERIMENTS[@]}"; do
        OUTPUT_PATH="$WORKSPACE/results/$DATASET/$EXPERIMENT_NAME.parquet"
        python3 -m verl.trainer.amo_eval \
            data.path=$OUTPUT_PATH \
            custom_reward_function.path=$REWARD_FUNCTION_PATH 
    done
done