#!/bin/bash

set -x

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")

DATASET_NAME="PKU-SafeRLHF"
MODEL_NAMES=(
    "Qwen2.5-7B-SafeRLHF-RM"
    "Qwen2.5-7B-SafeRLHF-CM"
)
SPLIT="test"

for MODEL_NAME in ${MODEL_NAMES[@]}
do
    REWARD_MODEL_PATH=$WORKSPACE/playground/reward_model/checkpoints/$MODEL_NAME

    python3 $WORKSPACE/playground/reward_calibration/build_calibration.py \
        --model_path $REWARD_MODEL_PATH \
        --dataset_path $WORKSPACE/playground/reward_model \
        --dataset_name $DATASET_NAME \
        --split $SPLIT \
        --output_path $WORKSPACE/playground/reward_calibration/config/${MODEL_NAME}_${SPLIT}_calibration.json \
        --p 0.1
done