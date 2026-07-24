#!/usr/bin/env bash

configure_dataset() {
    PROJECT_NAME=amo_math-lighteval
    RESULTS_DATASET=MATH-LightEval
    TRAIN_FILES=$WORKSPACE/data/MATH-LightEval/train.parquet
    VAL_FILES=$WORKSPACE/data/MATH-LightEval/test.parquet
    REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_math/math_accuracy.py','$WORKSPACE/recipe/amo_math/math_conciseness.py','$WORKSPACE/recipe/amo_math/math_format.py']"

    PROFILE_EPOCHS=50
    PROFILE_TRAIN_BATCH_SIZE=512
    PROFILE_MAX_PROMPT_LENGTH=2048
    PROFILE_MAX_RESPONSE_LENGTH=2048
    PROFILE_PPO_MINI_BATCH_SIZE=128
    PROFILE_SAVE_FREQ=10
    PROFILE_TEST_FREQ=10

    case "$MODEL" in
        qwen2.5-1.5b|qwen2.5-3b) PROFILE_MICRO_BATCH_SIZE_PER_GPU=32 ;;
        llama3.2-3b) PROFILE_MICRO_BATCH_SIZE_PER_GPU=16 ;;
    esac
}
