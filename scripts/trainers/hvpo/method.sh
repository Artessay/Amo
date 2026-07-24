#!/usr/bin/env bash

configure_method() {
    ADV_ESTIMATOR=hvpo
    REWARD_MANAGER=amo_hvpo

    case "$TRAINER_VARIANT" in
        "") ;;
        distance)
            VARIANT_OVERRIDES+=("amo_strategy.hv_config.distance_metric=none")
            ;;
        euclidean)
            VARIANT_OVERRIDES+=("amo_strategy.hv_config.distance_metric=euclidean")
            ;;
        lag1)
            PROFILE_MICRO_BATCH_SIZE_PER_GPU=16
            PROFILE_TEST_FREQ=1
            ;;
        lag3)
            PROFILE_MICRO_BATCH_SIZE_PER_GPU=16
            PROFILE_TEST_FREQ=3
            ;;
        lag7)
            PROFILE_MICRO_BATCH_SIZE_PER_GPU=16
            PROFILE_TEST_FREQ=7
            ;;
        *)
            die "unknown HVPO variant '$TRAINER_VARIANT'"
            ;;
    esac
}
