#!/bin/bash

set -x

WORKSPACE=$(dirname "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")")
PORT=50053

python3 $WORKSPACE/recipe/amo_news/summarization_server.py \
    --port $PORT
