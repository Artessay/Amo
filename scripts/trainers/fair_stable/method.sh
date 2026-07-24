#!/usr/bin/env bash

configure_method() {
    [[ -z $TRAINER_VARIANT ]] || die "Fair-and-Stable does not define variants"
    ADV_ESTIMATOR=grpo
    REWARD_MANAGER=amo_adaptive
    METHOD_OVERRIDES=(
        "amo_strategy.adaptive_config.method=fair_stable"
        "amo_strategy.adaptive_config.weight_lr=0.1"
    )
}
