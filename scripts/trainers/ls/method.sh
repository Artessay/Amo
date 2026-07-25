#!/usr/bin/env bash

configure_method() {
    ADV_ESTIMATOR=grpo
    REWARD_MANAGER=amo_scalarize
    METHOD_OVERRIDES=("amo_strategy.scalarize_config.method=linear")

    if [[ -n $TRAINER_VARIANT ]]; then
        configure_h2_weight_variant "amo_strategy.scalarize_config.weights" "LS"
    fi

    if [[ $DATASET == pku-saferlhf ]]; then
        load_safe_calibration
        METHOD_OVERRIDES+=(
            "amo_strategy.scalarize_config.normalize=affine"
            "amo_strategy.scalarize_config.calib_lower=$SAFE_CALIB_LOWER"
            "amo_strategy.scalarize_config.calib_upper=$SAFE_CALIB_UPPER"
        )
    fi
}
