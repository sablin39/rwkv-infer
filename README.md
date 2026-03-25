# RWKV7 External SGLang Integration

This repository contains the external-package integration for serving RWKV7VL with SGLang without modifying upstream `sglang` source.

> [!NOTE]  
> `sglang` sticks at `transformers==4.57.1` due to RoPE errors. However our RWKV-VL uses encoder from Qwen3.5, so a transformes at 5.x is required. You will need to manually install it via `pip install -U transformers`

## Layout

- `src/rwkv7vl_model`
  Implements the external SGLang model entrypoint `RWKV7VLForConditionalGeneration`.
  The model keeps the HF-compatible composition of Qwen3.5 vision encoder, projector, RWKV7 language model, and LM head.
  It uses SGLang-managed Mamba-style recurrent cache storage so RWKV state can participate in prefix cache reuse.

- `src/rwkv7vl_processor`
  Implements the external multimodal processor `RWKV7VLImageProcessor`.
  The processor handles image loading, tokenization, and multimodal bookkeeping for RWKV7VL prompts.

- `src/rwkv7_backend`
  Holds the reusable RWKV7 recurrent-cache and block implementation shared by the external model package.

## Runtime Notes

- The external model package patches SGLang at import time in a narrow, RWKV7VL-specific way.
- Those patches let SGLang recognize configs that expose `mamba2_cache_params`, enable the recurrent cache path used by RWKV7VL, and track reusable prefix states for cache reuse.
- The processor package stays lightweight so the tokenizer worker can register the model architecture name without importing the full model stack.

## Launch

Use the external-package path with the HuggingFace export in `./ckpt/rwkv-23552`:

```bash
export SGLANG_DISABLE_CUDNN_CHECK=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_EXTERNAL_MODEL_PACKAGE=rwkv7vl_model
export SGLANG_EXTERNAL_MM_MODEL_ARCH=RWKV7VLForConditionalGeneration
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=rwkv7vl_processor

./.venv/bin/python -m sglang.launch_server \
  --model-path ./ckpt/rwkv-23552 \
  --trust-remote-code \
  --port 30000
```

The same launch flow is also wrapped in `./benchmarks/launch_server.sh`.

## RWKV7 Backend Selection

RWKV7 recurrent execution now supports split backend selection for prefill and decode.
The default remains `fla` for both phases.

Environment variables:

- `RWKV7_BACKEND`
  Shared fallback for both phases.
- `RWKV7_PREFILL_BACKEND`
  Prefill-only override.
- `RWKV7_DECODE_BACKEND`
  Decode-only override.

Supported values:

- `fla`
- `jit`

Examples:

```bash
# Use the JIT backend for both prefill and decode.
RWKV7_BACKEND=jit ./benchmarks/launch_server.sh

# Keep prefill on FLA but switch decode to JIT.
RWKV7_PREFILL_BACKEND=fla RWKV7_DECODE_BACKEND=jit ./benchmarks/launch_server.sh

# Use JIT for prefill and keep decode on FLA.
RWKV7_PREFILL_BACKEND=jit RWKV7_DECODE_BACKEND=fla ./benchmarks/launch_server.sh
```

## Benchmarks

The repository includes a small set of ready-to-run helpers under `./benchmarks` for the core server flow: launch the server, drive batched requests, and optionally profile that same serving path. They default to:

- `MODEL_PATH=./ckpt/rwkv-23552`
- `TOKENIZER_PATH=./ckpt/rwkv-23552`
- `HOST=127.0.0.1`
- `PORT=31000`
- `PYTHONPATH=./src:$PYTHONPATH`
- `DATASET_NAME=random`
- `GSP_NUM_TURNS=4` when `DATASET_NAME=generated-shared-prefix`
- the external RWKV7VL package environment variables required by SGLang

If you add more exports under `./ckpt`, override `MODEL_PATH` and `TOKENIZER_PATH` when launching the server or benchmarks.

Run the server first:

```bash
./benchmarks/launch_server.sh
```

Then use the benchmark scripts from the repo root.

- `./benchmarks/bench_serving.sh`
  Runs the online serving benchmark used for repeated batched requests.
  By default it uses `DATASET_NAME=random` as a lightweight text-only baseline.
  Set `DATASET_NAME=generated-shared-prefix` to exercise shared-prefix, multi-round conversations with `GSP_NUM_TURNS=4` by default.
  Set `DATASET_NAME=image` for synthetic image-text traffic or `DATASET_NAME=mmmu` for a real image-text benchmark loaded from Hugging Face.

- `./benchmarks/profile_serving.sh`
  Wraps the same online serving benchmark with `--profile`, so traces come from the exact workload selected via `DATASET_NAME` instead of a separate microbenchmark.
  The launcher exports `SGLANG_TORCH_PROFILER_DIR` by default, so profiling works out of the box when the server is started with `./benchmarks/launch_server.sh`.

- `./benchmarks/bench_rwkv7_jit_ops.py`
  Directly compares the repo-local RWKV7 JIT recurrent kernels against the current FLA recurrent ops for prefill and decode.
  This is a focused smoke benchmark for the new backend rather than an end-to-end server benchmark.

Examples:

```bash
# Default random-text serving benchmark.
./benchmarks/bench_serving.sh

# Multi-round shared-prefix serving benchmark.
DATASET_NAME=generated-shared-prefix ./benchmarks/bench_serving.sh

# Increase turns on the shared-prefix benchmark.
DATASET_NAME=generated-shared-prefix GSP_NUM_TURNS=8 ./benchmarks/bench_serving.sh

# Synthetic image-text traffic.
DATASET_NAME=image IMAGE_COUNT=1 IMAGE_RESOLUTION=360p ./benchmarks/bench_serving.sh

# Real image-text prompts from MMMU Math.
DATASET_NAME=mmmu NUM_PROMPTS=32 ./benchmarks/bench_serving.sh

# Profile the same multi-round workload and save traces elsewhere.
DATASET_NAME=generated-shared-prefix PROFILE_OUTPUT_DIR=./profiles/rwkv7vl ./benchmarks/profile_serving.sh
```

Most scripts accept extra CLI flags and common environment overrides. For example:

```bash
PORT=32000 MODEL_PATH=/path/to/ckpt/rwkv-23552 ./benchmarks/launch_server.sh --log-requests
PORT=32000 ./benchmarks/profile_serving.sh --profile-stages prefill decode
```

Notes:

- `DATASET_NAME=image` uses synthetic images generated by SGLang for multimodal load testing.
- `DATASET_NAME=mmmu` downloads the `MMMU/MMMU` Math test split from Hugging Face on first use.

## Source Of Truth

- Model weights, config, tokenizer, processor assets, and chat template come from `./ckpt/rwkv-23552`.
- The external package is focused on runtime adaptation for SGLang serving, not checkpoint or tokenizer format changes.
