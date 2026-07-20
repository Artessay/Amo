#!/bin/bash

set -x

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")

# sentence-transformers/LaBSE
MODEL_PATH=$WORKSPACE/playground/detox_model/LaBSE
PORT=50061

python3 $WORKSPACE/recipe/amo_detox/sim_server.py \
    --model_path $MODEL_PATH \
    --port $PORT
