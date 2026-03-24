import logging

import torch

logger = logging.getLogger(__name__)
_RWKV7VL_ARCH = "RWKV7VLForConditionalGeneration"


class _RWKV7Mamba2ConfigAdapter:
    """ModelRunner-facing view over configs that only advertise mamba2 cache params.

    SGLang's hybrid recurrent memory profiler currently assumes a non-empty
    `full_attention_layer_ids` list when sizing per-token KV cache. RWKV7VL is
    fully recurrent, so the underlying HF config correctly reports no full
    attention layers. We keep that source-of-truth config untouched and expose a
    minimal runner-only adapter here so ModelRunner can finish sizing its hybrid
    pools without upstream SGLang edits.
    """

    def __init__(self, config):
        self._config = config

    def __getattr__(self, name):
        return getattr(self._config, name)

    @property
    def full_attention_layer_ids(self):
        layer_ids = list(getattr(self._config, "full_attention_layer_ids", []) or [])
        return layer_ids if layer_ids else [0]

    @property
    def linear_layer_ids(self):
        return getattr(self._config, "linear_layer_ids", [])


def _should_defer_patch() -> bool:
    get_default_device = getattr(torch, "get_default_device", None)
    if get_default_device is None:
        return False
    try:
        return torch.device(get_default_device()).type == "meta"
    except Exception:
        return False


def _should_patch_rwkv7vl(batch_like) -> bool:
    model_config = getattr(batch_like, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    architectures = list(getattr(hf_config, "architectures", []) or [])
    return _RWKV7VL_ARCH in architectures or getattr(hf_config, "model_type", None) == "modrwkv"


def force_rwkv7vl_extra_buffer() -> None:
    try:
        from sglang.srt.server_args import get_global_server_args
    except Exception as exc:
        logger.debug("RWKV7VL could not import global server args yet: %s", exc)
        return

    try:
        server_args = get_global_server_args()
    except Exception as exc:
        logger.debug("RWKV7VL could not read global server args yet: %s", exc)
        return

    if server_args is None:
        return

    if getattr(server_args, "mamba_scheduler_strategy", None) != "extra_buffer":
        logger.info(
            "Promoting RWKV7VL to SGLang mamba extra_buffer mode so recurrent prefix cache can reuse prompt prefixes."
        )
        server_args.mamba_scheduler_strategy = "extra_buffer"


def patch_schedule_batch_tracking() -> None:
    try:
        from sglang.srt.managers.schedule_batch import ScheduleBatch
    except Exception as exc:
        logger.warning("Failed to import ScheduleBatch for RWKV7VL patch: %s", exc)
        return

    patch_flag = "_rwkv7vl_track_patch_applied"
    if getattr(ScheduleBatch, patch_flag, False):
        return

    original = ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend

    def _patched(self, req, mamba_track_mask_cpu, mamba_track_indices_cpu, mamba_track_seqlens_cpu):
        if not _should_patch_rwkv7vl(self):
            return original(
                self,
                req,
                mamba_track_mask_cpu,
                mamba_track_indices_cpu,
                mamba_track_seqlens_cpu,
            )

        prefix_len = len(req.prefix_indices)
        total_len = prefix_len + req.extend_input_len
        track_len = total_len - 1
        if (
            req.mamba_branching_seqlen is not None
            and prefix_len < req.mamba_branching_seqlen < total_len
        ):
            track_len = req.mamba_branching_seqlen

        mask = track_len > prefix_len
        mamba_track_mask_cpu.append(mask)
        mamba_track_indices_cpu.append(
            req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx].item()
        )

        if mask:
            req.mamba_next_track_idx = self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                req.mamba_next_track_idx
            )
            req.mamba_last_track_seqlen = track_len
            mamba_track_seqlens_cpu.append(track_len)
        else:
            req.mamba_last_track_seqlen = None
            mamba_track_seqlens_cpu.append(-1)

    ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend = _patched
    setattr(ScheduleBatch, patch_flag, True)
    logger.info("Patched ScheduleBatch mamba tracking for RWKV7VL exact reusable-prefix states")


def patch_mamba_attn_backend_tracking() -> None:
    try:
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            MambaAttnBackendBase,
        )
    except Exception as exc:
        logger.warning("Failed to import MambaAttnBackendBase for RWKV7VL patch: %s", exc)
        return

    patch_flag = "_rwkv7vl_mamba_backend_track_patch_applied"
    if getattr(MambaAttnBackendBase, patch_flag, False):
        return

    original_init = MambaAttnBackendBase.__init__
    original_forward_metadata = MambaAttnBackendBase._forward_metadata

    def _patched_init(self, model_runner):
        original_init(self, model_runner)
        self._rwkv7vl_disable_track_metadata = _should_patch_rwkv7vl(model_runner)

    def _patched_forward_metadata(self, forward_batch):
        if not getattr(self, "_rwkv7vl_disable_track_metadata", False):
            return original_forward_metadata(self, forward_batch)

        saved = (
            forward_batch.mamba_track_indices,
            forward_batch.mamba_track_mask,
            forward_batch.mamba_track_seqlens,
        )
        forward_batch.mamba_track_indices = None
        forward_batch.mamba_track_mask = None
        forward_batch.mamba_track_seqlens = None
        try:
            return original_forward_metadata(self, forward_batch)
        finally:
            (
                forward_batch.mamba_track_indices,
                forward_batch.mamba_track_mask,
                forward_batch.mamba_track_seqlens,
            ) = saved

    MambaAttnBackendBase.__init__ = _patched_init
    MambaAttnBackendBase._forward_metadata = _patched_forward_metadata
    setattr(MambaAttnBackendBase, patch_flag, True)
    logger.info("Patched Mamba attention backend metadata init for RWKV7VL tracking")


def patch_model_runner_mamba2_config() -> None:
    if _should_defer_patch():
        logger.debug("Deferring RWKV7VL ModelRunner patch while default device is meta.")
        return

    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
    except Exception as exc:
        logger.warning("Failed to import ModelRunner for RWKV7VL patch: %s", exc)
        return

    patch_flag = "_rwkv7vl_mamba2_config_patched"
    if getattr(ModelRunner, patch_flag, False):
        return

    original_property = ModelRunner.mamba2_config
    original_getter = original_property.fget

    def _patched_mamba2_config(self):
        result = original_getter(self)
        if result is not None:
            return result

        hf_config = self.model_config.hf_config
        try:
            cache_params = getattr(hf_config, "mamba2_cache_params", None)
        except Exception:
            return None
        if cache_params is None:
            return None
        return _RWKV7Mamba2ConfigAdapter(hf_config)

    ModelRunner.mamba2_config = property(_patched_mamba2_config)
    setattr(ModelRunner, patch_flag, True)
    logger.info("Patched ModelRunner.mamba2_config for RWKV7VL external model support")


force_rwkv7vl_extra_buffer()
patch_schedule_batch_tracking()
patch_mamba_attn_backend_tracking()
patch_model_runner_mamba2_config()
