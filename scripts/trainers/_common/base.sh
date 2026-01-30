#!/usr/bin/env bash

configure_h2_weight_variant() {
    local override_key=$1
    local method_context=$2
    local objective_count
    case "$DATASET" in
        math-lighteval) objective_count=3 ;;
        news) objective_count=4 ;;
        pku-saferlhf|rlla) objective_count=2 ;;
        *) die "$method_context does not know the objective count for dataset '$DATASET'" ;;
    esac

    [[ $TRAINER_VARIANT == h2w* ]] || \
        die "invalid $method_context variant '$TRAINER_VARIANT' (expected h2w<digits>)"
    local encoded=${TRAINER_VARIANT#h2w}
    [[ ${#encoded} -eq $objective_count ]] || \
        die "invalid $method_context variant '$TRAINER_VARIANT': dataset '$DATASET' needs $objective_count weight digits"
    [[ $encoded =~ ^[012]+$ ]] || \
        die "invalid $method_context variant '$TRAINER_VARIANT': weight digits must be 0, 1, or 2"

    local digit sum=0
    local -a weights=()
    local i
    for ((i=0; i<objective_count; i++)); do
        digit=${encoded:i:1}
        sum=$((sum + digit))
        case "$digit" in
            0) weights+=(0.0) ;;
            1) weights+=(0.5) ;;
            2) weights+=(1.0) ;;
        esac
    done
    [[ $sum -eq 2 ]] || \
        die "invalid $method_context variant '$TRAINER_VARIANT': H=2 weight digits must sum to 2"
    [[ $encoded != 11 ]] || \
        die "invalid $method_context variant '$TRAINER_VARIANT': the uniform centroid uses the base variant"

    local weights_csv=${weights[*]}
    weights_csv=${weights_csv// /,}
    VARIANT_OVERRIDES+=("$override_key=[$weights_csv]")
}

# Shared optimization/runtime defaults. Dataset, model, and method profiles may
# override these values before launch.sh resolves environment overrides.
configure_base() {
    PROFILE_EPOCHS=1
    PROFILE_ACTOR_LR=1e-5
    PROFILE_TRAIN_BATCH_SIZE=512
    PROFILE_MAX_PROMPT_LENGTH=2048
    PROFILE_MAX_RESPONSE_LENGTH=1024
    PROFILE_PPO_MINI_BATCH_SIZE=128
    PROFILE_MICRO_BATCH_SIZE_PER_GPU=32
    PROFILE_NUM_NODES=1
    PROFILE_NUM_GPUS_PER_NODE=2
    PROFILE_TENSOR_MODEL_PARALLEL_SIZE=1
    PROFILE_GPU_MEMORY_UTILIZATION=0.6
    PROFILE_ROLLOUT_N=4
    PROFILE_SAVE_FREQ=10
    PROFILE_TEST_FREQ=10
    PROFILE_RESUME_MODE=auto
    PROFILE_VAL_BEFORE_TRAIN=True
    PROFILE_TRAIN_GPUS=""
    PROFILE_LOGGER='["console", "swanlab"]'

    MODEL_OVERRIDES=()
    DATASET_OVERRIDES=()
    METHOD_OVERRIDES=()
    VARIANT_OVERRIDES=()
}
