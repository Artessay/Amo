#!/usr/bin/env bash

configure_method() {
    [[ -z $TRAINER_VARIANT ]] || die "NSGA-II-style credit does not define variants"
    ADV_ESTIMATOR=grpo
    REWARD_MANAGER=amo_pareto
    METHOD_OVERRIDES=("amo_strategy.pareto_config.method=nsga2")
}
