#!/bin/bash

set -x

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")

REWARD_MODEL_PATH=$WORKSPACE/playground/reward_model/checkpoints/Qwen2.5-7B-SafeRLHF-RM
PORT=50051

python reward_server.py --model_path $REWARD_MODEL_PATH --port $PORT
