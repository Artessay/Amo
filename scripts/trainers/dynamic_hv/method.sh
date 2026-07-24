#!/usr/bin/env bash

configure_method() {
    [[ -z $TRAINER_VARIANT ]] || die "Dynamic-HV does not define variants"
    ADV_ESTIMATOR=grpo
    REWARD_MANAGER=amo_adaptive
    METHOD_OVERRIDES=("amo_strategy.adaptive_config.method=dynamic_hv")

    if [[ $DATASET == pku-saferlhf ]]; then
        load_safe_calibration
        METHOD_OVERRIDES+=("amo_strategy.adaptive_config.hv_reference_point=$SAFE_HV_REFERENCE")
    fi
}
