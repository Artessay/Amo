#!/bin/bash

set -x

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")

# s-nlp/roberta_toxicity_classifier
MODEL_PATH=$WORKSPACE/playground/detox_model/roberta_toxicity_classifier
PORT=50060

python3 $WORKSPACE/recipe/amo_detox/sta_server.py \
    --model_path $MODEL_PATH \
    --port $PORT
