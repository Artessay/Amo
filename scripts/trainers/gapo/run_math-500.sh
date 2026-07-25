#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
exec bash "$SCRIPT_DIR/../_common/launch.sh" gapo math-500 "$@"
