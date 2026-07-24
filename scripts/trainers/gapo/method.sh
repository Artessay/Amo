#!/usr/bin/env bash

configure_method() {
    [[ -z $TRAINER_VARIANT ]] || die "GAPO does not define variants"
    ADV_ESTIMATOR=gapo
    REWARD_MANAGER=amo_vanilla
    METHOD_OVERRIDES=("algorithm.gapo_p=1.0")
}
