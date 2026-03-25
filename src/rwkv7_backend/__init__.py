from __future__ import annotations

import os

from fla.models.rwkv7 import RWKV7Config

from .cache import RWKV7LayerState, RWKV7SGLangCache
from .fla_backend import RWKV7Block

DEFAULT_RWKV7_BACKEND = "fla"
RWKV7FLABlock = RWKV7Block
RWKV7JITBlock = RWKV7Block

RWKV7_BACKEND_REGISTRY = {
    "fla": RWKV7FLABlock,
    "jit": RWKV7JITBlock,
}


def _normalize_rwkv7_backend_name(backend_name: str) -> str:
    normalized = backend_name.lower()
    if normalized not in RWKV7_BACKEND_REGISTRY:
        supported = ", ".join(sorted(RWKV7_BACKEND_REGISTRY))
        raise ValueError(
            f"Unsupported RWKV7 language backend '{backend_name}'. Supported backends: {supported}."
        )
    return normalized


def resolve_rwkv7_backend_name(config=None, backend_name: str | None = None) -> str:
    if backend_name is None and config is not None:
        backend_name = getattr(config, "language_model_backend", None)
    if backend_name is None and config is not None:
        backend_name = getattr(config, "rwkv7_backend", None)
    if backend_name is None:
        backend_name = os.environ.get("RWKV7_BACKEND", DEFAULT_RWKV7_BACKEND)
    return _normalize_rwkv7_backend_name(backend_name)


def resolve_rwkv7_phase_backend_name(
    phase: str,
    config=None,
    *,
    backend_name: str | None = None,
    phase_backend_name: str | None = None,
) -> str:
    if phase not in ("prefill", "decode"):
        raise ValueError(f"Unsupported RWKV7 phase '{phase}'. Expected 'prefill' or 'decode'.")

    if phase_backend_name is None and config is not None:
        phase_backend_name = getattr(config, f"rwkv7_{phase}_backend", None)
    if phase_backend_name is None:
        phase_backend_name = os.environ.get(f"RWKV7_{phase.upper()}_BACKEND")
    if phase_backend_name is None:
        phase_backend_name = resolve_rwkv7_backend_name(
            config=config,
            backend_name=backend_name,
        )
    return _normalize_rwkv7_backend_name(phase_backend_name)


def resolve_rwkv7_backend_names(
    config=None,
    *,
    backend_name: str | None = None,
    prefill_backend_name: str | None = None,
    decode_backend_name: str | None = None,
) -> tuple[str, str]:
    return (
        resolve_rwkv7_phase_backend_name(
            "prefill",
            config=config,
            backend_name=backend_name,
            phase_backend_name=prefill_backend_name,
        ),
        resolve_rwkv7_phase_backend_name(
            "decode",
            config=config,
            backend_name=backend_name,
            phase_backend_name=decode_backend_name,
        ),
    )


def get_rwkv7_block_class(
    config=None,
    backend_name: str | None = None,
    *,
    prefill_backend_name: str | None = None,
    decode_backend_name: str | None = None,
):
    resolved_prefill, resolved_decode = resolve_rwkv7_backend_names(
        config=config,
        backend_name=backend_name,
        prefill_backend_name=prefill_backend_name,
        decode_backend_name=decode_backend_name,
    )
    if resolved_prefill == resolved_decode:
        return RWKV7_BACKEND_REGISTRY[resolved_prefill]
    return RWKV7Block


def build_rwkv7_block(
    config: RWKV7Config,
    layer_idx: int,
    backend_name: str | None = None,
    *,
    prefill_backend_name: str | None = None,
    decode_backend_name: str | None = None,
):
    resolved_prefill, resolved_decode = resolve_rwkv7_backend_names(
        config=config,
        backend_name=backend_name,
        prefill_backend_name=prefill_backend_name,
        decode_backend_name=decode_backend_name,
    )
    block_cls = get_rwkv7_block_class(
        config=config,
        backend_name=backend_name,
        prefill_backend_name=resolved_prefill,
        decode_backend_name=resolved_decode,
    )
    return block_cls(
        config,
        layer_idx,
        prefill_backend_name=resolved_prefill,
        decode_backend_name=resolved_decode,
    )


__all__ = [
    "DEFAULT_RWKV7_BACKEND",
    "RWKV7_BACKEND_REGISTRY",
    "RWKV7FLABlock",
    "RWKV7JITBlock",
    "RWKV7Block",
    "RWKV7LayerState",
    "RWKV7SGLangCache",
    "build_rwkv7_block",
    "get_rwkv7_block_class",
    "resolve_rwkv7_backend_name",
    "resolve_rwkv7_backend_names",
    "resolve_rwkv7_phase_backend_name",
]
