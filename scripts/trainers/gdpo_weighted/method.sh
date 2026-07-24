#!/usr/bin/env bash

configure_method() {
    [[ -z $TRAINER_VARIANT ]] || die "weighted GDPO does not define variants"
    ADV_ESTIMATOR=gdpo_weighted
    REWARD_MANAGER=amo_vanilla
}
