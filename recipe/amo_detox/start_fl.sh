#!/bin/bash

set -x

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")

# textattack/roberta-base-CoLA
MODEL_PATH=$WORKSPACE/playground/detox_model/roberta-base-CoLA
PORT=50062

python3 $WORKSPACE/recipe/amo_detox/fl_server.py \
    --model_path $MODEL_PATH \
    --port $PORT
