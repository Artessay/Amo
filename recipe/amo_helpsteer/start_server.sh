#!/bin/bash

set -x

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")

# Path or HuggingFace id of the ArmoRM reward model.
# Override REWARD_MODEL_PATH to point at a local checkpoint if desired.
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-RLHFlow/ArmoRM-Llama3-8B-v0.1}
PORT=${PORT:-50054}

python3 $WORKSPACE/recipe/amo_helpsteer/reward_server.py \
    --model_path $REWARD_MODEL_PATH \
    --port $PORT
