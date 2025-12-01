set -e

CHECKPOINT_DIR="./outputs/Qwen3-0.6B-SafeRLHF-RM/slice_end"
OUTPUT_DIR="./checkpoints/Qwen3-0.6B-SafeRLHF-RM"
# ---
# CHECKPOINT_DIR="./outputs/Qwen3-0.6B-SafeRLHF-CM/slice_end"
# OUTPUT_DIR="./checkpoints/Qwen3-0.6B-SafeRLHF-CM"
# ---
# CHECKPOINT_DIR="./outputs/Qwen3-4B-SafeRLHF-RM/slice_end"
# OUTPUT_DIR="./checkpoints/Qwen3-4B-SafeRLHF-RM"


python zero_to_fp32.py ${CHECKPOINT_DIR} ${OUTPUT_DIR} --safe_serialization
cp ${CHECKPOINT_DIR}/*.json ${OUTPUT_DIR}/
cp ${CHECKPOINT_DIR}/*.jinja ${OUTPUT_DIR}/
