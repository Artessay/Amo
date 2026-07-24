#!/usr/bin/env bash

configure_method() {
    [[ -z $TRAINER_VARIANT ]] || die "Lagrangian does not define variants"
    ADV_ESTIMATOR=grpo
    REWARD_MANAGER=amo_adaptive
    METHOD_OVERRIDES=(
        "amo_strategy.adaptive_config.method=lagrangian"
        "amo_strategy.adaptive_config.primary_index=0"
        "amo_strategy.adaptive_config.lambda_lr=0.05"
    )

    case "$DATASET" in
        math-lighteval)
            METHOD_OVERRIDES+=("amo_strategy.adaptive_config.budgets=[0.0,0.5,0.5]")
            ;;
        pku-saferlhf)
            load_safe_calibration
            METHOD_OVERRIDES+=("amo_strategy.adaptive_config.budgets=[0.0,$SAFE_HARMLESS_BUDGET]")
            ;;
        *)
            die "Lagrangian has no calibrated defaults for dataset '$DATASET'"
            ;;
    esac
}
