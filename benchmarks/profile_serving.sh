#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

cmd=(
  "$(dirname "$0")/bench_serving.sh"
  --profile
  --profile-output-dir "${PROFILE_OUTPUT_DIR:-${REPO_ROOT}/sglang_profile}"
  --profile-num-steps "${PROFILE_NUM_STEPS:-20}"
  --profile-prefix "${PROFILE_PREFIX:-serving}"
)

if [[ "${PROFILE_BY_STAGE:-1}" != "0" ]]; then
  cmd+=(--profile-by-stage)
fi

exec "${cmd[@]}" "$@"
