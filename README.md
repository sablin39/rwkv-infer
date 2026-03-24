# RWKV7VL External SGLang Integration

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

## Benchmarks

The repository includes ready-to-run benchmark helpers under `./benchmarks`. They default to:

- `MODEL_PATH=./ckpt/rwkv-23552`
- `TOKENIZER_PATH=./ckpt/rwkv-23552`
- `HOST=127.0.0.1`
- `PORT=31000`
- `PYTHONPATH=./src:$PYTHONPATH`
- the external RWKV7VL package environment variables required by SGLang

If you add more exports under `./ckpt`, override `MODEL_PATH` and `TOKENIZER_PATH` when launching the server or benchmarks.

Run the server first:

```bash
./benchmarks/launch_server.sh
```

Then use the benchmark scripts from the repo root.

- `./benchmarks/bench_cache_reuse.sh`
  Stress-tests shared-prefix cache reuse with `generated-shared-prefix`. This is the most relevant built-in test for validating recurrent cache reuse and radix-cache behavior.
  The script intentionally does not pass `--tokenize-prompt`, because the current `sglang.bench_serving` implementation asserts that `generated-shared-prefix` stays in text-prompt mode.

- `./benchmarks/bench_random_throughput.sh`
  Measures text-only serving throughput and concurrency without shared prefixes. Use it as the baseline against the cache-reuse benchmark.

- `./benchmarks/bench_cache_hit_rate.sh`
  Runs `bench_one_batch_server` against the live server with a configurable synthetic cache hit rate. Set `CACHE_HIT_RATE=0.0` and `CACHE_HIT_RATE=0.9` to estimate the payoff from cache reuse.

- `./benchmarks/profile_prefill.sh`
  Profiles one-batch prefill latency across several batch sizes and prompt lengths.

- `./benchmarks/profile_decode.sh`
  Profiles decode latency and logs per-step timing so stalls in the recurrent-state path are easier to spot.

- `./benchmarks/profile_server.sh`
  Captures CPU, GPU, and memory traces from the running SGLang server into `./sglang_profile`.

- `./benchmarks/bench_image_serving.sh`
  Stress-tests the multimodal serving path with synthetic image requests.

- `./benchmarks/bench_offline_gsp.sh`
  Launches an offline throughput run with shared prefixes for cold-start or isolated throughput measurements.

Examples:

```bash
# Compare shared-prefix reuse against random-text throughput.
./benchmarks/bench_cache_reuse.sh
./benchmarks/bench_random_throughput.sh

# Compare no-cache and high-cache synthetic hit rates.
CACHE_HIT_RATE=0.0 ./benchmarks/bench_cache_hit_rate.sh
CACHE_HIT_RATE=0.9 ./benchmarks/bench_cache_hit_rate.sh

# Increase concurrency on the random-text throughput benchmark.
MAX_CONCURRENCY=128 REQUEST_RATE=32 ./benchmarks/bench_random_throughput.sh

# Save server traces to a custom output directory.
OUTPUT_DIR=./profiles/rwkv7vl ./benchmarks/profile_server.sh
```

Most scripts accept extra CLI flags and common environment overrides. For example:

```bash
PORT=32000 MODEL_PATH=/path/to/ckpt/rwkv-23552 ./benchmarks/launch_server.sh --log-requests
PORT=32000 ./benchmarks/bench_cache_reuse.sh --profile --profile-output-dir ./profiles/cache
```

## Source Of Truth

- Model weights, config, tokenizer, processor assets, and chat template come from `./ckpt/rwkv-23552`.
- The external package is focused on runtime adaptation for SGLang serving, not checkpoint or tokenizer format changes.
