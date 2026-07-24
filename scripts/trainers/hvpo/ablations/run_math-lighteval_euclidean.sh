#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
export TRAINER_VARIANT=euclidean
exec bash "$SCRIPT_DIR/../../_common/launch.sh" hvpo math-lighteval "$@"
