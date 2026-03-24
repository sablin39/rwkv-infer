# RWKV7VL External SGLang Integration

This repository contains the external-package integration for serving RWKV7VL with SGLang without modifying upstream `sglang` source.

## Layout

- `rwkv7vl_model`
  Implements the external SGLang model entrypoint `RWKV7VLForConditionalGeneration`.
  The model keeps the HF-compatible composition of Qwen3.5 vision encoder, projector, RWKV7 language model, and LM head.
  It uses SGLang-managed Mamba-style recurrent cache storage so RWKV state can participate in prefix cache reuse.

- `rwkv7vl_processor`
  Implements the external multimodal processor `RWKV7VLImageProcessor`.
  The processor handles image loading, tokenization, and multimodal bookkeeping for RWKV7VL prompts.

## Runtime Notes

- The external model package patches SGLang at import time in a narrow, RWKV7VL-specific way.
- Those patches let SGLang recognize configs that expose `mamba2_cache_params`, enable the recurrent cache path used by RWKV7VL, and track reusable prefix states for cache reuse.
- The processor package stays lightweight so the tokenizer worker can register the model architecture name without importing the full model stack.

## Launch

Use the external-package path with the HuggingFace export in `./rwkv-23552`:

```bash
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=rwkv7vl_model
export SGLANG_EXTERNAL_MM_MODEL_ARCH=RWKV7VLForConditionalGeneration
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=rwkv7vl_processor

./.venv/bin/python -m sglang.launch_server \
  --model-path ./rwkv-23552 \
  --trust-remote-code \
  --port 30000
```

## Source Of Truth

- Model weights, config, tokenizer, processor assets, and chat template come from `./rwkv-23552`.
- The external package is focused on runtime adaptation for SGLang serving, not checkpoint or tokenizer format changes.
