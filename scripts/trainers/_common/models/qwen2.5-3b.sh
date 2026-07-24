#!/usr/bin/env bash

configure_model() {
    MODEL_TAG=qwen2.5-3b
    MODEL_PATH=${AMO_MODEL_PATH:-/data/Qwen/Qwen2.5-3B-Instruct}
    MODEL_OVERRIDES=(
        "actor_rollout_ref.model.lora_rank=32"
        "actor_rollout_ref.model.lora_alpha=16"
    )
}
