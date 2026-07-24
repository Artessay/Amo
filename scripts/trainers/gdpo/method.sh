#!/usr/bin/env bash

configure_method() {
    [[ -z $TRAINER_VARIANT ]] || die "GDPO does not define variant '$TRAINER_VARIANT'"
    ADV_ESTIMATOR=gdpo
    REWARD_MANAGER=amo_vanilla
}
