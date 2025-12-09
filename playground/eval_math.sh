set -x

WORKSPACE=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
echo "Using workspace: $WORKSPACE"

DATASET="MATH-500"

EXPERIMENTS=(
    "qwen3-4b"
    "qwen3-4b_grpo"
    "qwen3-4b_vanilla"
)

REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_math/math_accuracy.py','$WORKSPACE/recipe/amo_math/math_conciseness.py','$WORKSPACE/recipe/amo_math/math_format.py']"

# Evaluation
for EXPERIMENT_NAME in "${EXPERIMENTS[@]}"; do
    python3 -m verl.trainer.amo_eval \
        data.path=$WORKSPACE/results/$DATASET/$EXPERIMENT_NAME.parquet \
        custom_reward_function.path=$REWARD_FUNCTION_PATH 
done
