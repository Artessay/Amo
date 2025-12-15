set -e

# Define model sizes and types to loop through
MODEL_NAMES=(
    "Qwen2.5-7B"
    # "Qwen3-0.6B"
    # "Qwen3-1.7B" 
    # "Qwen3-4B" 
    # "Qwen3-8B"
)
DATA_NAMES=(
    "SafeRLHF"
)
MODEL_TYPES=(
    # "RM"
    "CM"
)

# Loop through all models and types
for DATA_NAME in "${DATA_NAMES[@]}"; do
    for MODEL_NAME in "${MODEL_NAMES[@]}"; do
        for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
            # Compose directory names
            DIR_NAME="${MODEL_NAME}-${DATA_NAME}-${MODEL_TYPE}"
            CHECKPOINT_DIR="./outputs/${DIR_NAME}/slice_end"
            OUTPUT_DIR="./checkpoints/${DIR_NAME}"

            # Print info (optional)
            echo "Processing ${DIR_NAME}..."

            # Run conversion script
            python zero_to_fp32.py "${CHECKPOINT_DIR}" "${OUTPUT_DIR}" --safe_serialization

            # Copy json and jinja files
            cp "${CHECKPOINT_DIR}"/*.json "${OUTPUT_DIR}/"
            cp "${CHECKPOINT_DIR}"/*.jinja "${OUTPUT_DIR}/"
        done
    done
done