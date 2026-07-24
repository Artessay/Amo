#!/usr/bin/env bash

configure_method() {
    [[ -z $TRAINER_VARIANT ]] || die "GRPO does not define variant '$TRAINER_VARIANT'"
    ADV_ESTIMATOR=grpo
    REWARD_MANAGER=amo_vanilla
}
