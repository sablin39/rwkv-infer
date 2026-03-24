#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" exec "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP:-1}" \
  "$@"
