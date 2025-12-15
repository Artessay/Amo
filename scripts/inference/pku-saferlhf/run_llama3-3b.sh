set -x

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")
echo "Using workspace: $WORKSPACE"

EXPERIMENT_NAME="llama3-3b"

MODEL_PATH="/data/meta-llama/Llama-3.2-3B-Instruct"
DATA_PATH="$WORKSPACE/data/PKU-SafeRLHF/test.parquet"
OUTPUT_PATH="$WORKSPACE/results/PKU-SafeRLHF/${EXPERIMENT_NAME}.parquet"

GEN_SCRIPT_PATH="$WORKSPACE/playground/inference/generation.py"


python3 $GEN_SCRIPT_PATH \
    --data $DATA_PATH \
    --model $MODEL_PATH \
    --output $OUTPUT_PATH