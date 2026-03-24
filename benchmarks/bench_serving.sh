#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/common.sh"

dataset_name="${DATASET_NAME:-random}"
backend="${BACKEND:-sglang-oai-chat}"
cmd=(
  "${PYTHON_BIN}" -m sglang.bench_serving
  --backend "${backend}"
  --base-url "${BASE_URL}"
  --dataset-name "${dataset_name}"
  --tokenizer "${TOKENIZER_PATH}"
)

case "${dataset_name}" in
  generated-shared-prefix)
    gsp_num_groups="${GSP_NUM_GROUPS:-10}"
    gsp_prompts_per_group="${GSP_PROMPTS_PER_GROUP:-20}"
    cmd+=(
      --num-prompts "${NUM_PROMPTS:-$((gsp_num_groups * gsp_prompts_per_group))}"
      --request-rate "${REQUEST_RATE:-8}"
      --max-concurrency "${MAX_CONCURRENCY:-32}"
      --warmup-requests "${WARMUP_REQUESTS:-10}"
      --gsp-num-groups "${gsp_num_groups}"
      --gsp-prompts-per-group "${gsp_prompts_per_group}"
      --gsp-system-prompt-len "${GSP_SYSTEM_PROMPT_LEN:-512}"
      --gsp-question-len "${GSP_QUESTION_LEN:-64}"
      --gsp-output-len "${GSP_OUTPUT_LEN:-32}"
      --gsp-num-turns "${GSP_NUM_TURNS:-4}"
      --gsp-range-ratio "${GSP_RANGE_RATIO:-1.0}"
    )
    if [[ "${FLUSH_CACHE:-1}" != "0" ]]; then
      cmd+=(--flush-cache)
    fi
    if [[ "${GSP_FAST_PREPARE:-0}" != "0" ]]; then
      cmd+=(--gsp-fast-prepare)
    fi
    if [[ "${GSP_ORDERED:-0}" != "0" ]]; then
      cmd+=(--gsp-ordered)
    fi
    if [[ "${GSP_SEND_ROUTING_KEY:-0}" != "0" ]]; then
      cmd+=(--gsp-send-routing-key)
    fi
    ;;
  random)
    cmd+=(
      --num-prompts "${NUM_PROMPTS:-500}"
      --request-rate "${REQUEST_RATE:-16}"
      --max-concurrency "${MAX_CONCURRENCY:-64}"
      --warmup-requests "${WARMUP_REQUESTS:-20}"
      --random-input-len "${RANDOM_INPUT_LEN:-256}"
      --random-output-len "${RANDOM_OUTPUT_LEN:-64}"
      --random-range-ratio "${RANDOM_RANGE_RATIO:-1.0}"
    )
    if [[ "${TOKENIZE_PROMPT:-1}" != "0" ]]; then
      cmd+=(--tokenize-prompt)
    fi
    ;;
  image)
    cmd+=(
      --num-prompts "${NUM_PROMPTS:-100}"
      --request-rate "${REQUEST_RATE:-4}"
      --max-concurrency "${MAX_CONCURRENCY:-8}"
      --warmup-requests "${WARMUP_REQUESTS:-10}"
      --random-input-len "${RANDOM_INPUT_LEN:-64}"
      --random-output-len "${RANDOM_OUTPUT_LEN:-32}"
      --random-range-ratio "${RANDOM_RANGE_RATIO:-1.0}"
      --image-count "${IMAGE_COUNT:-1}"
      --image-resolution "${IMAGE_RESOLUTION:-360p}"
      --image-format "${IMAGE_FORMAT:-jpeg}"
      --image-content "${IMAGE_CONTENT:-random}"
    )
    if [[ "${RANDOM_IMAGE_COUNT:-0}" != "0" ]]; then
      cmd+=(--random-image-count)
    fi
    ;;
  mmmu)
    cmd+=(
      --num-prompts "${NUM_PROMPTS:-100}"
      --request-rate "${REQUEST_RATE:-2}"
      --max-concurrency "${MAX_CONCURRENCY:-8}"
      --warmup-requests "${WARMUP_REQUESTS:-5}"
      --random-output-len "${MMMU_OUTPUT_LEN:-${RANDOM_OUTPUT_LEN:-256}}"
    )
    ;;
  *)
    echo "Unsupported DATASET_NAME=${dataset_name}. Use random, generated-shared-prefix, image, or mmmu." >&2
    exit 1
    ;;
esac

exec "${cmd[@]}" "$@"
