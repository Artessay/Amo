set -x

WORKSPACE=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
echo "Using workspace: $WORKSPACE"

DATASET="RLLA" 

EXPERIMENTS=(
    # "qwen2.5-1.5b"
    #"qwen2.5-1.5b_grpo"
    #"qwen2.5-1.5b_gdpo"
    # "qwen2.5-1.5b_hv"
    
    # "qwen2.5-3b"
    "qwen2.5-3b_grpo"
    "qwen2.5-3b_gdpo"

    # "llama3-3b"
    #"llama3.2-3b_grpo"
    #"llama3.2-3b_gdpo"
)

REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_tool/tool_correctness.py','$WORKSPACE/recipe/amo_tool/tool_format.py']"

# Evaluation
for EXPERIMENT_NAME in "${EXPERIMENTS[@]}"; do
    OUTPUT_PATH="$WORKSPACE/results/$DATASET/$EXPERIMENT_NAME.parquet"
    python3 -m verl.trainer.amo_eval \
        data.path=$OUTPUT_PATH \
        custom_reward_function.path=$REWARD_FUNCTION_PATH 
done