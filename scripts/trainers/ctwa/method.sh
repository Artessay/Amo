#!/usr/bin/env bash

configure_method() {
    [[ -z $TRAINER_VARIANT ]] || die "CTWA does not define variants"
    ADV_ESTIMATOR=grpo
    REWARD_MANAGER=amo_adaptive
    METHOD_OVERRIDES=(
        "amo_strategy.adaptive_config.method=ctwa"
        "amo_strategy.adaptive_config.weight_lr=0.1"
    )

    case "$DATASET" in
        math-lighteval) METHOD_OVERRIDES+=("amo_strategy.adaptive_config.cov_targets=[0.0,0.0,0.0]") ;;
        news) METHOD_OVERRIDES+=("amo_strategy.adaptive_config.cov_targets=[0.0,0.0,0.0,0.0]") ;;
        pku-saferlhf) METHOD_OVERRIDES+=("amo_strategy.adaptive_config.cov_targets=[0.0,0.0]") ;;
        rlla) METHOD_OVERRIDES+=("amo_strategy.adaptive_config.cov_targets=[0.0,0.0]") ;;
        *) die "CTWA has no covariance target defaults for dataset '$DATASET'" ;;
    esac
}
