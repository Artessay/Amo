#!/usr/bin/env bash

configure_method() {
    ADV_ESTIMATOR=hvpo
    REWARD_MANAGER=amo_hvpo
    METHOD_OVERRIDES=(
        "algorithm.hvpo_adv_scale_ema_decay=0.99"
        "algorithm.hvpo_adv_scale_epsilon=1.0e-6"
        "algorithm.hvpo_adv_scale_initial=1.0"
    )

    case "$DATASET" in
        pku-saferlhf)
            load_safe_calibration
            METHOD_OVERRIDES+=(
                "amo_strategy.hv_config.calib_lower=$SAFE_CALIB_LOWER"
                "amo_strategy.hv_config.calib_upper=$SAFE_CALIB_UPPER"
                "amo_strategy.hv_config.reference_point=[0,0]"
            )
            ;;
        math-lighteval|math-500)
            METHOD_OVERRIDES+=("amo_strategy.hv_config.calib_lower=[0,0,0]" "amo_strategy.hv_config.calib_upper=[1,1,1]" "amo_strategy.hv_config.reference_point=[0,0,0]")
            ;;
        paradetox)
            # style-transfer accuracy and fluency are probabilities; LaBSE
            # semantic similarity is a cosine in [-1, 1].
            METHOD_OVERRIDES+=("amo_strategy.hv_config.calib_lower=[0,-1,0]" "amo_strategy.hv_config.calib_upper=[1,1,1]" "amo_strategy.hv_config.reference_point=[0,0,0]")
            ;;
        news)
            METHOD_OVERRIDES+=("amo_strategy.hv_config.calib_lower=[0,0,0,0]" "amo_strategy.hv_config.calib_upper=[1,1,1,1]" "amo_strategy.hv_config.reference_point=[0,0,0,0]")
            ;;
        rlla)
            METHOD_OVERRIDES+=("amo_strategy.hv_config.calib_lower=[0,0]" "amo_strategy.hv_config.calib_upper=[1,1]" "amo_strategy.hv_config.reference_point=[0,0]")
            ;;
        *) die "HVPO has no fixed calibration for dataset '$DATASET'" ;;
    esac

    case "$TRAINER_VARIANT" in
        "") ;;
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
