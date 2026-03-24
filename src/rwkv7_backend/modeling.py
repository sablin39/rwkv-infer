from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.rwkv6 import LoRA
from fla.models.rwkv7 import RWKV7Config
from fla.modules import GroupNorm, LayerNorm
from fla.modules.activations import ACT2FN
from fla.modules.l2norm import l2_norm
from fla.modules.token_shift import token_shift
from fla.ops.rwkv7 import chunk_rwkv7, fused_mul_recurrent_rwkv7
from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7
from fla.ops.rwkv7.fused_k_update import fused_k_rwkv7
from fla.ops.rwkv7.gate_output_correction import gate_output_correction

from .cache import RWKV7LayerState, RWKV7SGLangCache

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer


class RWKV7Attention(nn.Module):
    def __init__(
        self,
        mode: str = "chunk",
        hidden_size: int = 1024,
        head_dim: int | None = 64,
        num_heads: int | None = None,
        decay_low_rank_dim: int | None = None,
        gate_low_rank_dim: int | None = None,
        a_low_rank_dim: int | None = None,
        v_low_rank_dim: int | None = None,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-5,
        layer_idx: int | None = None,
        fuse_norm: bool = False,
        value_dim: int | None = None,
        num_hidden_layers: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs

        self.mode = mode
        if mode not in ["chunk", "fused_recurrent"]:
            raise ValueError(f"Not supported mode `{mode}`.")

        self.hidden_size = hidden_size
        self.key_dim = hidden_size
        self.value_dim = value_dim if value_dim is not None else hidden_size
        if head_dim is None and num_heads is None:
            raise ValueError("Either `head_dim` or `num_heads` must be specified.")
        if head_dim is not None:
            self.head_dim = head_dim
            self.num_heads = int(hidden_size // head_dim)
        else:
            self.head_dim = int(hidden_size // num_heads)
            self.num_heads = int(num_heads)
        self.head_v_dim = int(self.value_dim // self.num_heads)

        factor = self.head_dim / 64
        self.decay_low_rank_dim = (
            decay_low_rank_dim
            if decay_low_rank_dim is not None
            else max(
                32,
                int(round((2.5 * (hidden_size**0.5)) * factor / 32) * 32),
            )
        )
        self.gate_low_rank_dim = (
            gate_low_rank_dim
            if gate_low_rank_dim is not None
            else max(32, int(round((5 * (hidden_size**0.5)) / 32) * 32))
        )
        self.a_low_rank_dim = (
            a_low_rank_dim
            if a_low_rank_dim is not None
            else max(
                32,
                int(round((2.5 * (hidden_size**0.5)) * factor / 32) * 32),
            )
        )
        self.v_low_rank_dim = (
            v_low_rank_dim
            if v_low_rank_dim is not None
            else max(
                32,
                int(round((1.7 * (hidden_size**0.5)) * factor / 32) * 32),
            )
        )

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        self.fuse_norm = fuse_norm

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_r = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.k_k = nn.Parameter(torch.zeros(self.key_dim))
        self.k_a = nn.Parameter(torch.zeros(self.key_dim))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        self.r_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.w_lora = LoRA(
            hidden_size,
            self.key_dim,
            low_rank_dim=self.decay_low_rank_dim,
            activation="tanh",
        )
        if self.layer_idx != 0:
            self.v_lora = LoRA(
                hidden_size,
                self.value_dim,
                low_rank_dim=self.v_low_rank_dim,
                activation=None,
            )
        self.a_lora = LoRA(
            hidden_size,
            self.key_dim,
            low_rank_dim=self.a_low_rank_dim,
            activation=None,
        )
        self.g_lora = LoRA(
            hidden_size,
            self.value_dim,
            low_rank_dim=self.gate_low_rank_dim,
            activation="sigmoid",
            bias=False,
        )

        if self.fuse_norm:
            self.g_norm = GroupNorm(
                num_groups=self.num_heads,
                hidden_size=self.value_dim,
                elementwise_affine=elementwise_affine,
                eps=self.head_dim * norm_eps,
                bias=True,
            )
        else:
            self.g_norm = nn.GroupNorm(
                num_groups=self.num_heads,
                num_channels=self.value_dim,
                eps=self.head_dim * norm_eps,
                affine=elementwise_affine,
            )

        try:
            from transformers.modeling_utils import _init_weights
        except ImportError:
            _init_weights = True
        if _init_weights:
            self.apply(self._initialize_weights)
        for _, module in self.named_modules():
            module._in_rwkv_module = True

    @torch.no_grad()
    @torch.compiler.disable
    def _initialize_weights(self, module: nn.Module) -> None:
        if getattr(module, "_is_hf_initialized", False):
            return

        if isinstance(module, RWKV7Attention) and self.layer_idx is not None:
            ratio_0_to_1 = self.layer_idx / (self.num_hidden_layers - 1)
            ratio_1_to_almost0 = 1.0 - (self.layer_idx / self.num_hidden_layers)

            ddd = torch.ones(1, 1, self.hidden_size, device=self.x_r.device)
            www = torch.zeros(self.hidden_size, device=self.x_r.device)
            zigzag = torch.zeros(self.hidden_size, device=self.x_r.device)
            linear = torch.zeros(self.hidden_size, device=self.x_r.device)
            for n in range(self.hidden_size):
                linear[n] = n / (self.hidden_size - 1) - 0.5
                zigzag[n] = ((n % self.head_dim) - ((self.head_dim - 1) / 2)) / (
                    (self.head_dim - 1) / 2
                )
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (self.hidden_size - 1)) ** (
                    1 + ratio_0_to_1**0.3
                )
                ddd[0, 0, n] = n / self.hidden_size

            self.x_r.data = (
                1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)
            ).to(self.x_r.dtype)
            self.x_w.data = (
                1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)
            ).to(self.x_w.dtype)
            self.x_k.data = (
                1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)
            ).to(self.x_k.dtype)
            self.x_v.data = (
                1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)
            ).to(self.x_v.dtype)
            self.x_a.data = (
                1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)
            ).to(self.x_a.dtype)
            self.x_g.data = (
                1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)
            ).to(self.x_g.dtype)

            nn.init.constant_(self.k_a, 1.02)
            nn.init.constant_(self.r_k, -0.04)
            self.k_k.data.copy_(
                (torch.zeros(self.hidden_size, device=self.k_k.device) + 0.71 - linear * 0.1).to(
                    self.k_k.dtype
                )
            )
            self.w_lora.set_bias_value(www + 0.5 + zigzag * 2.5)
            self.a_lora.set_bias_value(-0.19 + zigzag * 0.3 + linear * 0.4)

            if self.layer_idx != 0:
                self.v_lora._initialize_weights(self.v_lora)
                self.v_lora.set_bias_value(0.73 - linear * 0.4)

            self.g_norm.weight.data[:] = (
                (self.layer_idx + 1) / self.num_hidden_layers
            ) ** 0.7

            self._orthogonal_init(self.r_proj.weight)
            self._orthogonal_init(self.k_proj.weight, gain=0.1)
            self._orthogonal_init(self.v_proj.weight)
            self.o_proj.weight.data.zero_()

            del ddd, www, zigzag, linear

        module._is_hf_initialized = True

    @staticmethod
    def _orthogonal_init(weight: torch.Tensor, gain: float = 1.0) -> None:
        original_dtype = weight.dtype
        weight_fp32 = weight.float()
        nn.init.orthogonal_(weight_fp32, gain=gain)
        weight.data.copy_(weight_fp32.to(original_dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        state: RWKV7LayerState | None = None,
        attention_mask: torch.Tensor | None = None,
        v_first: torch.Tensor | None = None,
        cu_seqlens: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, RWKV7LayerState | None, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError(
                    "Expected attention_mask with shape [batch_size, seq_len]."
                )
            am = attention_mask.narrow(
                1, attention_mask.size(1) - seq_len, seq_len
            ).unsqueeze(-1)
        else:
            am = None

        if am is not None:
            hidden_states = hidden_states.mul(am)

        if state is None:
            conv_cache = None
            recurrent_state = None
        else:
            conv_cache = state.attn_shift_state
            recurrent_state = state.recurrent_state

        delta, conv_state = token_shift(
            hidden_states, cu_seqlens, output_cache=True, cache=conv_cache
        )
        xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(
            hidden_states,
            delta,
            self.x_r,
            self.x_w,
            self.x_k,
            self.x_v,
            self.x_a,
            self.x_g,
        )

        r = self.r_proj(xr)
        w = -0.6065306597126334 * self.w_lora(xw).sigmoid()

        k = self.k_proj(xk)
        v = self.v_proj(xv)

        if self.layer_idx == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())
        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        if self.fuse_norm:
            kk = l2_norm(
                rearrange(k * self.k_k, "b t (h d) -> b t h d", d=self.head_dim)
            )
        else:
            kk = F.normalize(
                rearrange(k * self.k_k, "b t (h d) -> b t h d", d=self.head_dim),
                dim=-1,
                p=2.0,
            )

        k = fused_k_rwkv7(k, a, self.k_a)

        if am is not None:
            v = v * am

        r, w, k, a = map(
            lambda x: rearrange(x, "b t (h d) -> b t h d", d=self.head_dim),
            (r, w, k, a),
        )
        v = rearrange(v, "b t (h d) -> b t h d", d=self.head_v_dim)

        if self.training or seq_len >= 64:
            o, recurrent_state = chunk_rwkv7(
                r=r,
                w=w,
                k=k,
                v=v,
                a=-kk,
                b=kk * a,
                scale=1.0,
                initial_state=recurrent_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                safe_gate=True,
                chunk_size=64,
            )
        else:
            o, recurrent_state = fused_mul_recurrent_rwkv7(
                r=r,
                w=w,
                k=k,
                v=v,
                kk=kk,
                a=a,
                scale=1.0,
                initial_state=recurrent_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
            )

        if self.fuse_norm:
            o = self.g_norm(rearrange(o, "... h d -> ... (h d)"))
        else:
            o = self.g_norm(rearrange(o, "b t h d -> (b t) (h d)")).view(
                batch_size, seq_len, -1
            )

        o = gate_output_correction(o, r, k, self.r_k, v, g)
        o = self.o_proj(o)

        if state is None:
            state = RWKV7LayerState(
                recurrent_state=recurrent_state,
                attn_shift_state=conv_state,
                ffn_shift_state=None,
            )
        else:
            state.recurrent_state = recurrent_state
            state.attn_shift_state = conv_state

        return o, state, v_first


class RWKV7FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: int | None = None,
        intermediate_size: int | None = None,
        hidden_act: str = "sqrelu",
        layer_idx: int | None = None,
        num_hidden_layers: int | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio)
            intermediate_size = 32 * ((intermediate_size + 32 - 1) // 32)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.zeros(hidden_size))
        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        try:
            from transformers.modeling_utils import _init_weights
        except ImportError:
            _init_weights = True
        if _init_weights:
            self.apply(self._initialize_weights)
        for _, module in self.named_modules():
            module._in_rwkv_module = True

    def _initialize_weights(self, module: nn.Module) -> None:
        if not isinstance(module, RWKV7FeedForward):
            return

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (module.layer_idx / module.num_hidden_layers)
            ddd = torch.ones(1, 1, module.hidden_size)
            for i in range(module.hidden_size):
                ddd[0, 0, i] = i / module.hidden_size
            module.x_k.data = 1.0 - torch.pow(ddd, ratio_1_to_almost0**4).squeeze()

        original_dtype = module.key.weight.dtype
        module.key.weight.data = nn.init.orthogonal_(
            module.key.weight.data.to(torch.float32)
        ).to(original_dtype)
        module.value.weight.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: RWKV7LayerState | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, RWKV7LayerState | None]:
        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2] :, None])

        delta, ffn_state = token_shift(
            x,
            cu_seqlens,
            cache=None if state is None else state.ffn_shift_state,
            output_cache=True,
        )

        if state is None:
            state = RWKV7LayerState(
                recurrent_state=None,
                attn_shift_state=None,
                ffn_shift_state=ffn_state,
            )
        else:
            state.ffn_shift_state = ffn_state

        return self.value(self.act_fn(self.key(x.addcmul(delta, self.x_k)))), state


class RWKV7Block(GradientCheckpointingLayer):
    def __init__(self, config: RWKV7Config, layer_idx: int) -> None:
        super().__init__()

        if config.attn is not None and layer_idx in config.attn["layers"]:
            raise NotImplementedError(
                "RWKV7VL's SGLang backend only supports pure recurrent RWKV7 layers."
            )

        self.config = config
        self.layer_idx = layer_idx

        if config.norm_first and layer_idx == 0:
            self.pre_norm = (LayerNorm if config.fuse_norm else nn.LayerNorm)(
                config.hidden_size,
                bias=config.norm_bias,
                eps=config.norm_eps,
            )
        self.attn_norm = (LayerNorm if config.fuse_norm else nn.LayerNorm)(
            config.hidden_size,
            bias=config.norm_bias,
            eps=config.norm_eps,
        )
        self.attn = RWKV7Attention(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            decay_low_rank_dim=config.decay_low_rank_dim,
            gate_low_rank_dim=config.gate_low_rank_dim,
            a_low_rank_dim=config.a_low_rank_dim,
            v_low_rank_dim=config.v_low_rank_dim,
            norm_eps=config.norm_eps,
            fuse_norm=config.fuse_norm,
            layer_idx=layer_idx,
            value_dim=config.value_dim[layer_idx],
            num_hidden_layers=config.num_hidden_layers,
        )
        self.ffn_norm = (LayerNorm if config.fuse_norm else nn.LayerNorm)(
            config.hidden_size,
            bias=config.norm_bias,
            eps=config.norm_eps,
        )
        self.ffn = RWKV7FeedForward(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            layer_idx=layer_idx,
            num_hidden_layers=config.num_hidden_layers,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cache: RWKV7SGLangCache | None = None,
        attention_mask: torch.Tensor | None = None,
        v_first: torch.Tensor | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs

        residual = (
            self.pre_norm(hidden_states) if hasattr(self, "pre_norm") else hidden_states
        )
        hidden_states = self.attn_norm(residual)

        layer_state = None if cache is None else cache.load(self.layer_idx)
        hidden_states, layer_state, v_first = self.attn(
            hidden_states,
            state=layer_state,
            attention_mask=attention_mask,
            v_first=v_first,
            cu_seqlens=cu_seqlens,
        )

        if self.config.fuse_norm:
            hidden_states, residual = self.ffn_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.ffn_norm(hidden_states)

        hidden_states, layer_state = self.ffn(
            hidden_states,
            state=layer_state,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
        )
        hidden_states = residual + hidden_states

        if cache is not None and layer_state is not None:
            cache.store(self.layer_idx, layer_state)

        return hidden_states, v_first
