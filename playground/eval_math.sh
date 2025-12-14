set -x

WORKSPACE=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
echo "Using workspace: $WORKSPACE"

# DATASET="MATH-500"
DATASET="MATH-lighteval"

EXPERIMENTS=(
    # "qwen3-4b"
    "qwen3-4b_grpo"
    # "qwen3-4b_vanilla"
    # "llama3-3b"
    # "qwen2.5-3b"
)

REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_math/math_accuracy.py','$WORKSPACE/recipe/amo_math/math_conciseness.py','$WORKSPACE/recipe/amo_math/math_format.py']"

# Evaluation
for EXPERIMENT_NAME in "${EXPERIMENTS[@]}"; do
    DATASET_NAME="${DATASET,,}"
    bash $WORKSPACE/scripts/inference/run_${EXPERIMENT_NAME}_${DATASET_NAME}.sh
    python3 -m verl.trainer.amo_eval \
        data.path=$WORKSPACE/results/$DATASET/$EXPERIMENT_NAME.parquet \
        custom_reward_function.path=$REWARD_FUNCTION_PATH 
done
