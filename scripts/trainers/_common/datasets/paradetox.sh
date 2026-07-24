#!/usr/bin/env bash

configure_dataset() {
    PROJECT_NAME=amo_paradetox
    RESULTS_DATASET=ParaDetox
    TRAIN_FILES=$WORKSPACE/data/ParaDetox/train.parquet
    VAL_FILES=$WORKSPACE/data/ParaDetox/test.parquet
    REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_detox/detox_sta.py','$WORKSPACE/recipe/amo_detox/detox_sim.py','$WORKSPACE/recipe/amo_detox/detox_fluency.py']"

    PROFILE_EPOCHS=1
    PROFILE_TRAIN_BATCH_SIZE=512
    PROFILE_MAX_PROMPT_LENGTH=512
    PROFILE_MAX_RESPONSE_LENGTH=512
    PROFILE_PPO_MINI_BATCH_SIZE=128
    PROFILE_MICRO_BATCH_SIZE_PER_GPU=32
    PROFILE_SAVE_FREQ=10
    PROFILE_TEST_FREQ=10
}
