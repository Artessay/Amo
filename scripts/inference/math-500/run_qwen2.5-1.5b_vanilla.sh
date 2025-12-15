set -x

WORKSPACE=$(dirname "$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")")
echo "Using workspace: $WORKSPACE"

PROJECT_NAME="amo_grpo_math-500"
EXPERIMENT_NAME="qwen2.5-1.5b_vanilla"

LATEST_STEP=$(cat checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/latest_checkpointed_iteration.txt)

DATA_PATH="$WORKSPACE/data/MATH-500/test.parquet"
MODEL_PATH="$WORKSPACE/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${LATEST_STEP}/actor/huggingface"
OUTPUT_PATH="$WORKSPACE/results/MATH-500/${EXPERIMENT_NAME}.parquet"

GEN_SCRIPT_PATH="$WORKSPACE/playground/inference/generation.py"


python3 $GEN_SCRIPT_PATH \
    --data $DATA_PATH \
    --model $MODEL_PATH \
    --output $OUTPUT_PATH