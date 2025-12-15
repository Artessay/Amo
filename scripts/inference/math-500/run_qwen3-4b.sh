set -x

WORKSPACE=$(dirname "$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")")
echo "Using workspace: $WORKSPACE"

EXPERIMENT_NAME="qwen3-4b"

MODEL_PATH="/data/Qwen/Qwen3-4B"
DATA_PATH="$WORKSPACE/data/MATH-500/test.parquet"
OUTPUT_PATH="$WORKSPACE/results/MATH-500/${EXPERIMENT_NAME}.parquet"

GEN_SCRIPT_PATH="$WORKSPACE/playground/inference/generation.py"


python3 $GEN_SCRIPT_PATH \
    --data $DATA_PATH \
    --model $MODEL_PATH \
    --output $OUTPUT_PATH