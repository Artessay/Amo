#!/bin/bash

set -x

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")

REWARD_MODEL_PATH=$WORKSPACE/playground/reward_model/checkpoints/Qwen2.5-7B-SafeRLHF-CM
PORT=50052

python $WORKSPACE/recipe/amo_safe/reward_server.py --model_path $REWARD_MODEL_PATH --port $PORT
