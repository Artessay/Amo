#!/usr/bin/env bash

load_safe_calibration() {
    local calibration_file=$WORKSPACE/results/PKU-SafeRLHF/safe_calibration.json
    local calibration_values=()
    [[ -f $calibration_file ]] || die "missing frozen calibration: $calibration_file"

    mapfile -t calibration_values < <(
        "$PYTHON_BIN" -c '
import json
import sys

with open(sys.argv[1]) as stream:
    data = json.load(stream)
compact = lambda value: json.dumps(value, separators=(",", ":"))
print(compact(data["calib_lower"]))
print(compact(data["calib_upper"]))
print(compact(data["ideal"]))
print(compact(data["hv_reference"]))
print(data["harmless_budget"])
' "$calibration_file"
    )
    ((${#calibration_values[@]} == 5)) || die "invalid calibration file: $calibration_file"
    SAFE_CALIB_LOWER=${calibration_values[0]}
    SAFE_CALIB_UPPER=${calibration_values[1]}
    SAFE_CALIB_IDEAL=${calibration_values[2]}
    SAFE_HV_REFERENCE=${calibration_values[3]}
    SAFE_HARMLESS_BUDGET=${calibration_values[4]}
}

configure_dataset() {
    PROJECT_NAME=amo_pku-saferlhf
    RESULTS_DATASET=PKU-SafeRLHF
    TRAIN_FILES=$WORKSPACE/data/PKU-SafeRLHF/train.parquet
    VAL_FILES=$WORKSPACE/data/PKU-SafeRLHF/test.parquet
    REWARD_FUNCTION_PATH="['$WORKSPACE/recipe/amo_safe/safe_helpfulness.py','$WORKSPACE/recipe/amo_safe/safe_harmlessness.py']"

    PROFILE_EPOCHS=1
    PROFILE_TRAIN_BATCH_SIZE=512
    PROFILE_MAX_PROMPT_LENGTH=512
    PROFILE_MAX_RESPONSE_LENGTH=512
    PROFILE_PPO_MINI_BATCH_SIZE=128
    PROFILE_MICRO_BATCH_SIZE_PER_GPU=16
    PROFILE_GPU_MEMORY_UTILIZATION=0.5
    PROFILE_SAVE_FREQ=10
    PROFILE_TEST_FREQ=10
    PROFILE_VAL_BEFORE_TRAIN=False
    PROFILE_TRAIN_GPUS=0,1
}
