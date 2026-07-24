#!/usr/bin/env bash

configure_dataset() {
    PROJECT_NAME=amo_rlla
    RESULTS_DATASET=RLLA
    TRAIN_FILES=$WORKSPACE/data/RLLA/train.parquet
    VAL_FILES=$WORKSPACE/data/RLLA/test.parquet
    REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_tool/tool_correctness.py','$WORKSPACE/recipe/amo_tool/tool_format.py']"

    PROFILE_EPOCHS=15
    PROFILE_TRAIN_BATCH_SIZE=512
    PROFILE_MAX_PROMPT_LENGTH=2048
    PROFILE_MAX_RESPONSE_LENGTH=1024
    PROFILE_PPO_MINI_BATCH_SIZE=128
    PROFILE_SAVE_FREQ=10
    PROFILE_TEST_FREQ=10

    case "$MODEL" in
        qwen2.5-1.5b) PROFILE_MICRO_BATCH_SIZE_PER_GPU=32 ;;
        qwen2.5-3b) PROFILE_MICRO_BATCH_SIZE_PER_GPU=16 ;;
        llama3.2-3b) PROFILE_MICRO_BATCH_SIZE_PER_GPU=8 ;;
    esac
}
