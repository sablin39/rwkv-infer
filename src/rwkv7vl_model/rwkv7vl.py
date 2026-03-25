import logging
from typing import Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from sglang.srt.configs.qwen3_vl import Qwen3VLVisionConfig
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel

from fla.models.rwkv7 import RWKV7Config
from rwkv7_backend import (
    RWKV7SGLangCache,
    build_rwkv7_block,
    resolve_rwkv7_backend_names,
)

logger = logging.getLogger(__name__)


def _normalize_vision_config(vision_config) -> Qwen3VLVisionConfig:
    if isinstance(vision_config, Qwen3VLVisionConfig):
        config = vision_config
    else:
        if isinstance(vision_config, dict):
            raw_config = dict(vision_config)
        elif hasattr(vision_config, "to_dict"):
            raw_config = vision_config.to_dict()
        else:
            raw_config = {}

        for key in ("architectures", "model_type", "dtype"):
            raw_config.pop(key, None)
        raw_config.setdefault("deepstack_visual_indexes", [])
        config = Qwen3VLVisionConfig(**raw_config)

    if config.deepstack_visual_indexes is None:
        config.deepstack_visual_indexes = []
    return config


class VisualAdapter(nn.Module):
    """Project vision encoder outputs into RWKV embedding space."""

    def __init__(
        self,
        encoder_dim: int,
        project_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim or project_dim * 4
        self.pre_norm = nn.LayerNorm(project_dim)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, project_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x + self.pre_norm(x)


class RWKV7LanguageModel(nn.Module):
    """RWKV7 language model backed by SGLang-managed recurrent cache tensors."""

    def __init__(self, config: RWKV7Config):
        super().__init__()
        self.config = config
        self.prefill_backend_name, self.decode_backend_name = resolve_rwkv7_backend_names(config)
        self.backend_name = (
            self.prefill_backend_name
            if self.prefill_backend_name == self.decode_backend_name
            else f"{self.prefill_backend_name}/{self.decode_backend_name}"
        )
        if self.prefill_backend_name == self.decode_backend_name:
            self.config.language_model_backend = self.prefill_backend_name
        self.config.rwkv7_prefill_backend = self.prefill_backend_name
        self.config.rwkv7_decode_backend = self.decode_backend_name
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                build_rwkv7_block(
                    config,
                    layer_idx,
                    prefill_backend_name=self.prefill_backend_name,
                    decode_backend_name=self.decode_backend_name,
                )
                for layer_idx in range(self.num_layers)
            ]
        )

        use_fuse = getattr(config, "fuse_norm", False)
        norm_bias = getattr(config, "norm_bias", True)
        norm_eps = getattr(config, "norm_eps", 1e-5)
        if use_fuse:
            from fla.modules import LayerNorm

            self.norm = LayerNorm(config.hidden_size, bias=norm_bias, eps=norm_eps)
        else:
            self.norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=True, eps=norm_eps
            )

    def get_input_embeddings(self):
        return self.embeddings

    def _track_extend_prefix_states(
        self,
        *,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> None:
        track_mask = getattr(forward_batch, "mamba_track_mask", None)
        track_indices = getattr(forward_batch, "mamba_track_indices", None)
        track_seqlens = getattr(forward_batch, "mamba_track_seqlens", None)
        if track_mask is None or track_indices is None or track_seqlens is None:
            return

        track_mask_cpu = track_mask.detach().cpu().tolist()
        if not any(track_mask_cpu):
            return

        selected_req_indices = []
        selected_write_indices = []
        selected_prefix_lens = []
        selected_track_lens = []
        track_slices = []

        start = 0
        track_seqlens_cpu = track_seqlens.detach().cpu().tolist()
        for i, seq_len in enumerate(forward_batch.extend_seq_lens_cpu):
            prefix_len = forward_batch.extend_prefix_lens_cpu[i]
            actual_track_len = track_seqlens_cpu[i]
            if not track_mask_cpu[i] or actual_track_len is None or actual_track_len < 0:
                start += seq_len
                continue

            track_extend_len = actual_track_len - prefix_len
            if track_extend_len <= 0:
                start += seq_len
                continue

            selected_req_indices.append(i)
            selected_write_indices.append(track_indices[i].item())
            selected_prefix_lens.append(prefix_len)
            selected_track_lens.append(track_extend_len)
            track_slices.append(hidden_states[start : start + track_extend_len])
            start += seq_len

        if not selected_req_indices:
            return

        track_hidden_states = torch.cat(track_slices, dim=0).unsqueeze(0)
        cu_seqlens = torch.zeros(
            len(selected_track_lens) + 1,
            dtype=torch.long,
            device=hidden_states.device,
        )
        cu_seqlens[1:] = torch.as_tensor(
            selected_track_lens,
            dtype=torch.long,
            device=hidden_states.device,
        ).cumsum(0)

        selected_req_indices_tensor = torch.as_tensor(
            selected_req_indices,
            device=forward_batch.req_pool_indices.device,
            dtype=torch.long,
        )
        track_cache = RWKV7SGLangCache(
            req_to_token_pool=forward_batch.req_to_token_pool,
            req_pool_indices=forward_batch.req_pool_indices[selected_req_indices_tensor],
            write_indices=torch.as_tensor(
                selected_write_indices,
                device=forward_batch.req_pool_indices.device,
                dtype=torch.int64,
            ),
            extend_prefix_lens=selected_prefix_lens,
        )

        v_first = torch.zeros_like(track_hidden_states)
        for layer in self.layers:
            track_hidden_states, v_first = layer(
                track_hidden_states,
                cache=track_cache,
                v_first=v_first,
                cu_seqlens=cu_seqlens,
                backend_phase="prefill",
                **kwargs,
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if forward_batch is None:
            raise ValueError("RWKV7LanguageModel requires a ForwardBatch.")

        hidden_states = input_embeds if input_embeds is not None else self.embeddings(input_ids)

        is_extend = forward_batch.forward_mode.is_extend()
        is_decode = forward_batch.forward_mode.is_decode()
        if not (is_extend or is_decode):
            return hidden_states

        cache = RWKV7SGLangCache(
            req_to_token_pool=forward_batch.req_to_token_pool,
            req_pool_indices=forward_batch.req_pool_indices,
            extend_prefix_lens=forward_batch.extend_prefix_lens_cpu if is_extend else None,
            track_indices=getattr(forward_batch, "mamba_track_indices", None)
            if is_decode
            else None,
            track_mask=getattr(forward_batch, "mamba_track_mask", None)
            if is_decode
            else None,
        )

        if is_extend:
            backend_phase = "prefill"
            self._track_extend_prefix_states(
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                **kwargs,
            )
            seq_lens = forward_batch.extend_seq_lens_cpu
            cu_seqlens = torch.zeros(
                len(seq_lens) + 1,
                dtype=torch.long,
                device=hidden_states.device,
            )
            cu_seqlens[1:] = torch.as_tensor(
                seq_lens, dtype=torch.long, device=hidden_states.device
            ).cumsum(0)
            hidden_states = hidden_states.unsqueeze(0)
        else:
            backend_phase = "decode"
            cu_seqlens = None
            hidden_states = hidden_states.unsqueeze(1)

        v_first = torch.zeros_like(hidden_states)
        for layer in self.layers:
            hidden_states, v_first = layer(
                hidden_states,
                cache=cache,
                v_first=v_first,
                cu_seqlens=cu_seqlens,
                backend_phase=backend_phase,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        if is_extend:
            hidden_states = hidden_states.squeeze(0)
        else:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class RWKV7VLForConditionalGeneration(nn.Module):
    """SGLang-compatible RWKV7 vision-language model."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if quant_config is not None:
            raise NotImplementedError(
                "RWKV7VL external SGLang integration does not support quantized loading in v1."
            )
        del prefix

        text_config = config.text_config
        if isinstance(text_config, dict):
            text_config = RWKV7Config(**text_config)

        for attr in (
            "language_model_backend",
            "rwkv7_backend",
            "rwkv7_prefill_backend",
            "rwkv7_decode_backend",
        ):
            value = getattr(config, attr, None)
            if value is not None:
                setattr(text_config, attr, value)

        vision_config = _normalize_vision_config(config.vision_config)
        config.vision_config = vision_config
        proj_cfg = config.projector_config
        if isinstance(proj_cfg, dict):
            encoder_dim = proj_cfg["encoder_dim"]
            project_dim = proj_cfg["project_dim"]
            hidden_dim = proj_cfg.get("hidden_dim")
        else:
            encoder_dim = proj_cfg.encoder_dim
            project_dim = proj_cfg.project_dim
            hidden_dim = getattr(proj_cfg, "hidden_dim", None)

        for attr in ("vocab_size", "hidden_size", "num_hidden_layers"):
            if not hasattr(config, attr) or getattr(config, attr) is None:
                setattr(config, attr, getattr(text_config, attr))
        if (
            not hasattr(config, "num_attention_heads")
            or config.num_attention_heads is None
        ):
            config.num_attention_heads = getattr(text_config, "num_heads", 16)

        self.config = config
        self.vision_config = vision_config
        self.visual = Qwen3VLMoeVisionModel(vision_config)
        self.proj = VisualAdapter(encoder_dim, project_dim, hidden_dim)
        self.model = RWKV7LanguageModel(text_config)
        self.lm_head = nn.Linear(
            text_config.hidden_size, text_config.vocab_size, bias=False
        )
        self.logits_processor: Optional[LogitsProcessor] = None

        self.image_token_id = getattr(config, "image_token_id", 65532)
        self.vision_start_token_id = getattr(config, "vision_start_token_id", 65530)
        self.vision_end_token_id = getattr(config, "vision_end_token_id", 65531)

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        if not items:
            return torch.empty(
                0,
                self.config.hidden_size,
                device=next(self.proj.parameters()).device,
            )

        pixel_values = torch.cat([item.feature for item in items], dim=0)
        image_grid_thw = torch.cat([item.image_grid_thw for item in items], dim=0)

        vis_device = next(self.visual.parameters()).device
        vis_dtype = next(self.visual.parameters()).dtype
        pixel_values = pixel_values.to(device=vis_device, dtype=vis_dtype)
        # SGLang's Qwen3 vision path computes cu_seqlens from a NumPy view and
        # expects grid_thw to remain on CPU.
        image_grid_thw = image_grid_thw.to(device="cpu")

        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw)
        if isinstance(vision_output, torch.Tensor):
            vision_embeds = vision_output
        elif (
            hasattr(vision_output, "pooler_output")
            and vision_output.pooler_output is not None
        ):
            pooled_embeds = vision_output.pooler_output
            if pooled_embeds.shape[-1] == self.proj.mlp[0].in_features:
                vision_embeds = pooled_embeds
            else:
                vision_embeds = vision_output.last_hidden_state
        elif hasattr(vision_output, "last_hidden_state"):
            vision_embeds = vision_output.last_hidden_state
        else:
            vision_embeds = vision_output[0]

        merge = getattr(self.vision_config, "spatial_merge_size", 2)
        split_sizes = (image_grid_thw.prod(-1) // (merge**2)).tolist()
        embeds_list = torch.split(vision_embeds, split_sizes)

        projected = []
        for embeds in embeds_list:
            if embeds.dim() == 2:
                embeds = embeds.unsqueeze(0)
            projected_embeds = self.proj(embeds)
            projected.append(projected_embeds.reshape(-1, projected_embeds.shape[-1]))

        return torch.cat(projected, dim=0)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def _get_logits_processor(self) -> LogitsProcessor:
        if self.logits_processor is None:
            self.logits_processor = LogitsProcessor(self.config)
        return self.logits_processor

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        del positions, input_embeds, kwargs
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
        )

        return self._get_logits_processor()(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    @staticmethod
    def _map_weight_name(name: str) -> str:
        if name.startswith("model.encoder."):
            mapped_name = "visual." + name[len("model.encoder.") :]
            return mapped_name.replace(".attn.qkv.", ".attn.qkv_proj.")
        if name.startswith("model.proj."):
            return "proj." + name[len("model.proj.") :]
        if name.startswith("model.llm."):
            return "model." + name[len("model.llm.") :]
        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded: Set[str] = set()
        unexpected: List[Tuple[str, str]] = []

        for name, loaded_weight in weights:
            mapped_name = self._map_weight_name(name)
            if mapped_name not in params_dict:
                unexpected.append((name, mapped_name))
                continue

            param = params_dict[mapped_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded.add(mapped_name)

        if unexpected:
            sample = ", ".join(
                f"{orig}->{mapped}" for orig, mapped in unexpected[:8]
            )
            raise KeyError(
                "Unexpected checkpoint parameters encountered while loading RWKV7VL: "
                f"{sample}"
            )

        return loaded


EntryClass = RWKV7VLForConditionalGeneration
