from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class RWKV7LayerState:
    recurrent_state: Optional[torch.Tensor]
    attn_shift_state: Optional[torch.Tensor]
    ffn_shift_state: Optional[torch.Tensor]


class RWKV7SGLangCache:
    """Thin adapter over SGLang's recurrent state pool for RWKV7 layers."""

    def __init__(
        self,
        *,
        req_to_token_pool,
        req_pool_indices: torch.Tensor,
        write_indices: Optional[torch.Tensor] = None,
        extend_prefix_lens: Optional[List[int]] = None,
        track_indices: Optional[torch.Tensor] = None,
        track_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if not hasattr(req_to_token_pool, "get_mamba_indices") or not hasattr(
            req_to_token_pool, "mamba2_layer_cache"
        ):
            raise TypeError(
                "RWKV7VL requires SGLang's hybrid req-to-token pool with Mamba cache support."
            )

        self.req_to_token_pool = req_to_token_pool

        req_pool_indices = req_pool_indices.to(dtype=torch.int64)
        self.read_indices = req_to_token_pool.get_mamba_indices(req_pool_indices).to(
            dtype=torch.int64
        )
        self.write_indices = (
            write_indices.to(device=self.read_indices.device, dtype=torch.int64)
            if write_indices is not None
            else self.read_indices
        )

        self.write_targets = [self.write_indices]
        if track_indices is not None and track_mask is not None:
            track_indices = track_indices.to(
                device=self.read_indices.device, dtype=torch.int64
            )
            track_mask = track_mask.to(
                device=self.read_indices.device, dtype=torch.bool
            )
            # Avoid dynamic masked selections during CUDA graph capture by mapping
            # untracked rows back to their normal write slots.
            self.write_targets.append(
                torch.where(track_mask, track_indices, self.write_indices)
            )

        if extend_prefix_lens is None:
            self.prefix_mask = None
        else:
            prefix_mask = torch.as_tensor(
                [prefix_len > 0 for prefix_len in extend_prefix_lens],
                device=self.read_indices.device,
                dtype=torch.bool,
            )
            self.prefix_mask = None if bool(prefix_mask.all()) else prefix_mask

    def _layer_cache(self, layer_idx: int):
        return self.req_to_token_pool.mamba2_layer_cache(layer_idx)

    @staticmethod
    def _flatten_packed_conv(packed_conv: torch.Tensor) -> torch.Tensor:
        if packed_conv.dim() == 3:
            if packed_conv.shape[-1] != 1:
                raise RuntimeError(
                    f"Unexpected packed RWKV conv cache shape: {tuple(packed_conv.shape)}"
                )
            packed_conv = packed_conv.squeeze(-1)
        if packed_conv.dim() != 2 or packed_conv.shape[1] % 2 != 0:
            raise RuntimeError(
                f"Unexpected packed RWKV conv cache shape: {tuple(packed_conv.shape)}"
            )
        return packed_conv

    def load(self, layer_idx: int) -> RWKV7LayerState:
        layer_cache = self._layer_cache(layer_idx)

        packed_conv = self._flatten_packed_conv(
            layer_cache.conv[0].index_select(0, self.read_indices)
        )

        recurrent_state = layer_cache.temporal.index_select(0, self.read_indices)
        attn_shift_state, ffn_shift_state = packed_conv.chunk(2, dim=1)

        if self.prefix_mask is not None:
            prefix_mask = self.prefix_mask
            recurrent_state[~prefix_mask] = 0
            attn_shift_state[~prefix_mask] = 0
            ffn_shift_state[~prefix_mask] = 0

        return RWKV7LayerState(
            recurrent_state=recurrent_state,
            attn_shift_state=attn_shift_state,
            ffn_shift_state=ffn_shift_state,
        )

    def _write_all(self, target: torch.Tensor, values: Optional[torch.Tensor]) -> None:
        if values is None:
            return
        for indices in self.write_targets:
            target.index_copy_(0, indices, values)

    def store(self, layer_idx: int, state: RWKV7LayerState) -> None:
        layer_cache = self._layer_cache(layer_idx)

        packed_conv = None
        if (
            state.attn_shift_state is not None
            and state.ffn_shift_state is not None
        ):
            packed_conv = torch.cat(
                [state.attn_shift_state, state.ffn_shift_state], dim=1
            )
            if layer_cache.conv[0].dim() == 3:
                packed_conv = packed_conv.unsqueeze(-1)
            packed_conv = packed_conv.to(
                device=layer_cache.conv[0].device,
                dtype=layer_cache.conv[0].dtype,
            )

        recurrent_state = None
        if state.recurrent_state is not None:
            recurrent_state = state.recurrent_state.to(
                device=layer_cache.temporal.device,
                dtype=layer_cache.temporal.dtype,
            )

        self._write_all(layer_cache.conv[0], packed_conv)
        self._write_all(layer_cache.temporal, recurrent_state)
