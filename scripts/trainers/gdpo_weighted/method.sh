#!/usr/bin/env bash

configure_method() {
    ADV_ESTIMATOR=gdpo_weighted
    REWARD_MANAGER=amo_vanilla

    if [[ -n $TRAINER_VARIANT ]]; then
        configure_h2_weight_variant "algorithm.amo_objective_weights" "weighted GDPO"
    fi
}
