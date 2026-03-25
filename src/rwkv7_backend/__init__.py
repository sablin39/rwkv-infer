import os

from fla.models.rwkv7 import RWKV7Config

from .cache import RWKV7LayerState, RWKV7SGLangCache
from .fla_backend import RWKV7Block

DEFAULT_RWKV7_BACKEND = "fla"
RWKV7FLABlock = RWKV7Block

RWKV7_BACKEND_REGISTRY = {
    "fla": RWKV7FLABlock,
}


def resolve_rwkv7_backend_name(config=None, backend_name: str | None = None) -> str:
    if backend_name is None and config is not None:
        backend_name = getattr(config, "language_model_backend", None)
    if backend_name is None and config is not None:
        backend_name = getattr(config, "rwkv7_backend", None)
    if backend_name is None:
        backend_name = os.environ.get("RWKV7_BACKEND", DEFAULT_RWKV7_BACKEND)

    backend_name = backend_name.lower()
    if backend_name not in RWKV7_BACKEND_REGISTRY:
        supported = ", ".join(sorted(RWKV7_BACKEND_REGISTRY))
        raise ValueError(
            f"Unsupported RWKV7 language backend '{backend_name}'. Supported backends: {supported}."
        )
    return backend_name


def get_rwkv7_block_class(config=None, backend_name: str | None = None):
    resolved_backend = resolve_rwkv7_backend_name(
        config=config,
        backend_name=backend_name,
    )
    return RWKV7_BACKEND_REGISTRY[resolved_backend]


def build_rwkv7_block(
    config: RWKV7Config,
    layer_idx: int,
    backend_name: str | None = None,
):
    block_cls = get_rwkv7_block_class(config=config, backend_name=backend_name)
    return block_cls(config, layer_idx)


__all__ = [
    "DEFAULT_RWKV7_BACKEND",
    "RWKV7_BACKEND_REGISTRY",
    "RWKV7FLABlock",
    "RWKV7Block",
    "RWKV7LayerState",
    "RWKV7SGLangCache",
    "build_rwkv7_block",
    "get_rwkv7_block_class",
    "resolve_rwkv7_backend_name",
]
