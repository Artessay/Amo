#!/bin/bash

set -x

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")

# REWARD_MODEL_PATH=$WORKSPACE/playground/reward_model/checkpoints/Qwen2.5-7B-SafeRLHF-RM
REWARD_MODEL_PATH=/data/PKU-Alignment/beaver-7b-v3.0-reward
PORT=50051

python3 $WORKSPACE/recipe/amo_safe/reward_server.py \
    --model_path $REWARD_MODEL_PATH \
    --port $PORT 