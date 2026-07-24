#!/usr/bin/env bash

configure_method() {
    [[ -z $TRAINER_VARIANT ]] || die "MGDA does not define variants"
    ADV_ESTIMATOR=mgda
    REWARD_MANAGER=amo_vanilla
}
