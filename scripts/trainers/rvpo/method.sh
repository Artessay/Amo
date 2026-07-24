#!/usr/bin/env bash

configure_method() {
    [[ -z $TRAINER_VARIANT ]] || die "RVPO does not define variants"
    ADV_ESTIMATOR=rvpo
    REWARD_MANAGER=amo_vanilla
    METHOD_OVERRIDES=("algorithm.rvpo_k=1.0")
}
