#!/usr/bin/env bash
set -Eeuo pipefail

# Launch an OpenAI-compatible vLLM server (defaults: API_PORT=8000)
# Required envs: MODEL_PATH, MODEL_ID

load_env_file() {
  local env_file="${1:-.env}"
  [[ -f "$env_file" ]] || return 0

  # Export variables defined in .env (ignores comments and blank lines)
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
}

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || {
    printf 'ERROR: %s must be set via environment variables or .env\n' "$name" >&2
    printf 'Example: MODEL_PATH=/path/to/model MODEL_ID=model-name %s\n' "$0" >&2
    exit 1
  }
}

main() {
  load_env_file ".env"

  : "${API_PORT:=8000}"
  echo "Starting vLLM OpenAI API server on port ${API_PORT}..."

  require_env "MODEL_PATH"
  require_env "MODEL_ID"

  exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$API_PORT" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --enable-prefix-caching \
    --served-model-name "$MODEL_ID"
}

main "$@"