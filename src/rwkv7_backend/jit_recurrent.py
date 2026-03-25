from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from sglang.jit_kernel.utils import (
    DEFAULT_CFLAGS,
    DEFAULT_CUDA_CFLAGS,
    DEFAULT_INCLUDE,
    cache_once,
    make_cpp_args,
)
from tvm_ffi.cpp import load_inline

if TYPE_CHECKING:
    from tvm_ffi.module import Module


RWKV7_JIT_MAX_HEAD_DIM = 128
RWKV7_JIT_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)
_CSRC_DIR = Path(__file__).resolve().parent / "jit_csrc"
_KERNEL_HEADER = (_CSRC_DIR / "rwkv7_recurrent.cuh").resolve()


def _dtype_cpp_name(dtype: torch.dtype) -> str:
    return str(make_cpp_args(dtype))


def _build_module_source(head_dim: int, dtype: torch.dtype) -> str:
    dtype_name = _dtype_cpp_name(dtype)
    header_path = _KERNEL_HEADER.as_posix()
    return f"""
#include "{header_path}"

void rwkv7_prefill(
    tvm::ffi::TensorView r,
    tvm::ffi::TensorView w_logits,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView kk,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView seq_indptr,
    tvm::ffi::TensorView initial_state,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView final_state) {{
  rwkv7_backend::jit::RWKV7PrefillKernel<{head_dim}, {dtype_name}>::run(
      r, w_logits, k, v, kk, a, seq_indptr, initial_state, output, final_state);
}}

void rwkv7_decode(
    tvm::ffi::TensorView r,
    tvm::ffi::TensorView w_logits,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView kk,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView initial_state,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView final_state) {{
  rwkv7_backend::jit::RWKV7DecodeKernel<{head_dim}, {dtype_name}>::run(
      r, w_logits, k, v, kk, a, initial_state, output, final_state);
}}
"""


@cache_once
def _jit_rwkv7_module(head_dim: int, dtype: torch.dtype) -> Module:
    source = _build_module_source(head_dim, dtype)
    return load_inline(
        f"rwkv7_recurrent_{head_dim}_{_dtype_cpp_name(dtype)}",
        cuda_sources=[source],
        functions=["rwkv7_prefill", "rwkv7_decode"],
        extra_cflags=DEFAULT_CFLAGS,
        extra_cuda_cflags=[*DEFAULT_CUDA_CFLAGS, "--use_fast_math"],
        extra_include_paths=[*DEFAULT_INCLUDE, str(_CSRC_DIR)],
    )


def _ensure_jit_supported_tensor(name: str, tensor: torch.Tensor, *, head_dim: int) -> None:
    if not tensor.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if tensor.dtype not in RWKV7_JIT_SUPPORTED_DTYPES:
        supported = ", ".join(str(dtype) for dtype in RWKV7_JIT_SUPPORTED_DTYPES)
        raise RuntimeError(f"{name} dtype {tensor.dtype} is unsupported for RWKV7 JIT. Supported dtypes: {supported}")
    if not tensor.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous for RWKV7 JIT")
    if tensor.ndim != 4:
        raise RuntimeError(f"{name} must have shape [batch, seq, heads, dim], got {tuple(tensor.shape)}")
    if tensor.shape[-1] != head_dim:
        raise RuntimeError(f"{name} head_dim mismatch: expected {head_dim}, got {tensor.shape[-1]}")


def _validate_common_inputs(
    r: torch.Tensor,
    w_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
) -> tuple[int, int, int, int]:
    if r.shape != w_logits.shape or r.shape != k.shape or r.shape != v.shape or r.shape != kk.shape or r.shape != a.shape:
        raise RuntimeError(
            "RWKV7 JIT expects r, w_logits, k, v, kk, and a to share the same [batch, seq, heads, dim] shape"
        )

    batch_size, seq_len, num_heads, head_dim = r.shape
    if head_dim != v.shape[-1]:
        raise RuntimeError(
            f"RWKV7 JIT only supports square per-head state, got key dim {head_dim} and value dim {v.shape[-1]}"
        )
    if head_dim > RWKV7_JIT_MAX_HEAD_DIM:
        raise RuntimeError(
            f"RWKV7 JIT only supports head_dim <= {RWKV7_JIT_MAX_HEAD_DIM}, got {head_dim}"
        )

    for name, tensor in (
        ("r", r),
        ("w_logits", w_logits),
        ("k", k),
        ("v", v),
        ("kk", kk),
        ("a", a),
    ):
        _ensure_jit_supported_tensor(name, tensor, head_dim=head_dim)

    if not (
        r.device == w_logits.device == k.device == v.device == kk.device == a.device
    ):
        raise RuntimeError("RWKV7 JIT expects all input tensors to be on the same device")
    if not (
        r.dtype == w_logits.dtype == k.dtype == v.dtype == kk.dtype == a.dtype
    ):
        raise RuntimeError("RWKV7 JIT expects all input tensors to share the same dtype")

    return batch_size, seq_len, num_heads, head_dim


def _prepare_initial_state(
    initial_state: torch.Tensor | None,
    *,
    num_sequences: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    expected_shape = (num_sequences, num_heads, head_dim, head_dim)
    if initial_state is None:
        return torch.zeros(expected_shape, device=device, dtype=torch.float32)
    if not initial_state.is_cuda:
        raise RuntimeError("initial_state must be a CUDA tensor")
    if initial_state.dtype != torch.float32:
        raise RuntimeError(
            f"initial_state must use torch.float32 to match the RWKV recurrent cache, got {initial_state.dtype}"
        )
    if tuple(initial_state.shape) != expected_shape:
        raise RuntimeError(
            f"initial_state must have shape {expected_shape}, got {tuple(initial_state.shape)}"
        )
    if not initial_state.is_contiguous():
        raise RuntimeError("initial_state must be contiguous for RWKV7 JIT")
    if initial_state.device != device:
        raise RuntimeError("initial_state must be on the same device as the recurrent inputs")
    return initial_state


def _prepare_seq_indptr(
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
    if cu_seqlens is None:
        seq_indptr = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            seq_len,
            device=device,
            dtype=torch.int64,
        )
        return seq_indptr, batch_size

    if cu_seqlens.ndim != 1:
        raise RuntimeError(f"cu_seqlens must be 1D, got shape {tuple(cu_seqlens.shape)}")
    if batch_size != 1:
        raise RuntimeError(
            f"Variable-length RWKV7 JIT prefill expects batch size 1, got {batch_size}"
        )

    seq_indptr = cu_seqlens.to(device=device, dtype=torch.int64)
    if seq_indptr.numel() < 2:
        raise RuntimeError("cu_seqlens must contain at least two elements")
    total_tokens = batch_size * seq_len
    if int(seq_indptr[-1].item()) != total_tokens:
        raise RuntimeError(
            f"cu_seqlens[-1] must equal total flattened tokens {total_tokens}, got {int(seq_indptr[-1].item())}"
        )
    return seq_indptr.contiguous(), seq_indptr.numel() - 1


def run_rwkv7_prefill_jit(
    *,
    r: torch.Tensor,
    w_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, num_heads, head_dim = _validate_common_inputs(r, w_logits, k, v, kk, a)
    seq_indptr, num_sequences = _prepare_seq_indptr(
        batch_size=batch_size,
        seq_len=seq_len,
        device=r.device,
        cu_seqlens=cu_seqlens,
    )
    initial_state = _prepare_initial_state(
        initial_state,
        num_sequences=num_sequences,
        num_heads=num_heads,
        head_dim=head_dim,
        device=r.device,
    )

    output = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    module = _jit_rwkv7_module(head_dim, r.dtype)
    module.rwkv7_prefill(r, w_logits, k, v, kk, a, seq_indptr, initial_state, output, final_state)
    return output, final_state


def run_rwkv7_decode_jit(
    *,
    r: torch.Tensor,
    w_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, num_heads, head_dim = _validate_common_inputs(r, w_logits, k, v, kk, a)
    if seq_len != 1:
        raise RuntimeError(f"RWKV7 JIT decode expects sequence length 1, got {seq_len}")
    initial_state = _prepare_initial_state(
        initial_state,
        num_sequences=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        device=r.device,
    )

    output = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    module = _jit_rwkv7_module(head_dim, r.dtype)
    module.rwkv7_decode(r, w_logits, k, v, kk, a, initial_state, output, final_state)
    return output, final_state


__all__ = [
    "RWKV7_JIT_MAX_HEAD_DIM",
    "RWKV7_JIT_SUPPORTED_DTYPES",
    "run_rwkv7_decode_jit",
    "run_rwkv7_prefill_jit",
]
