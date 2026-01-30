#!/usr/bin/env bash

configure_model() {
    MODEL_TAG=qwen2.5-1.5b
    MODEL_PATH=${AMO_MODEL_PATH:-/data/Qwen/Qwen2.5-1.5B-Instruct}
    PROFILE_ACTOR_LR=2e-4
    MODEL_OVERRIDES=(
        "actor_rollout_ref.model.lora_rank=32"
        "actor_rollout_ref.model.lora_alpha=16"
    )
}
