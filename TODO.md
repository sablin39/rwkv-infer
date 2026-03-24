# TODO

## Utils

- [ ] JIT kernel modified from kernels in [rwkv7_fast_fused](https://github.com/BlinkDL/RWKV-CUDA/tree/main/rwkv7_fast_fused)

- [ ] Attention Backend selection for RWKV. 

- [ ] Huggingface-compatible exporters for RWKV and RWKV-VL.

- [ ] General benchmarks and testbenches, including cache reuse, TTFT,.etc. Refer to https://docs.sglang.io/developer_guide/benchmark_and_profiling.html

## RWKV-VL

- [ ] Use the [Qwen3VLMoeVisionModel](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_vl.py) instead of hf class.

- [ ] Verify if VLM scheduler feature is used. 