#!/usr/bin/env bash

configure_model() {
    MODEL_TAG=llama3.2-3b
    MODEL_PATH=${AMO_MODEL_PATH:-/data/meta-llama/Llama-3.2-3B-Instruct}
    # The existing GRPO/GDPO/HVPO Llama launchers train without LoRA overrides.
    MODEL_OVERRIDES=()
}
