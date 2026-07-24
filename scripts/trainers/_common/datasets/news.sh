#!/usr/bin/env bash

configure_dataset() {
    PROJECT_NAME=amo_cnn_dailymail
    RESULTS_DATASET=CNN_DailyMail
    TRAIN_FILES=$WORKSPACE/data/CNN_DailyMail/train.parquet
    VAL_FILES=$WORKSPACE/data/CNN_DailyMail/test.parquet
    REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_news/news_coherence.py','$WORKSPACE/recipe/amo_news/news_fluency.py','$WORKSPACE/recipe/amo_news/news_relevance.py','$WORKSPACE/recipe/amo_news/news_consistency.py']"

    PROFILE_TRAIN_BATCH_SIZE=512
    PROFILE_MAX_PROMPT_LENGTH=2048
    PROFILE_MAX_RESPONSE_LENGTH=1024
    PROFILE_PPO_MINI_BATCH_SIZE=128
    PROFILE_SAVE_FREQ=10
    PROFILE_TEST_FREQ=10

    case "$MODEL" in
        qwen2.5-1.5b)
            PROFILE_EPOCHS=15
            PROFILE_MICRO_BATCH_SIZE_PER_GPU=16
            ;;
        qwen2.5-3b)
            PROFILE_EPOCHS=15
            PROFILE_MICRO_BATCH_SIZE_PER_GPU=8
            ;;
        llama3.2-3b)
            PROFILE_EPOCHS=10
            PROFILE_MICRO_BATCH_SIZE_PER_GPU=16
            ;;
    esac
}
