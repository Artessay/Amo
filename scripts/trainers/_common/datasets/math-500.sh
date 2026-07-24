#!/usr/bin/env bash

configure_dataset() {
    PROJECT_NAME=amo_math-500
    RESULTS_DATASET=MATH-500
    TRAIN_FILES=$WORKSPACE/data/MATH-500/train.parquet
    VAL_FILES=$WORKSPACE/data/MATH-500/val.parquet
    REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_math/math_accuracy.py','$WORKSPACE/recipe/amo_math/math_conciseness.py','$WORKSPACE/recipe/amo_math/math_format.py']"

    PROFILE_EPOCHS=100
    PROFILE_TRAIN_BATCH_SIZE=100
    PROFILE_MAX_PROMPT_LENGTH=2048
    PROFILE_MAX_RESPONSE_LENGTH=2048
    PROFILE_PPO_MINI_BATCH_SIZE=50
    PROFILE_MICRO_BATCH_SIZE_PER_GPU=25
    PROFILE_SAVE_FREQ=20
    PROFILE_TEST_FREQ=20
}
