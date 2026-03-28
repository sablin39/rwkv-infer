import argparse
import os
import re
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from .configuration_rwkv7 import RWKV7Config
    from .modeling_rwkv7 import RWKV7ForCausalLM, RWKV7Model
    from .tokenizer import RwkvTokenizer
except ImportError:
    from configuration_rwkv7 import RWKV7Config
    from modeling_rwkv7 import RWKV7ForCausalLM, RWKV7Model
    from tokenizer import RwkvTokenizer


IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
IM_START_TOKEN_ID = 23
IM_END_TOKEN_ID = 24
DEFAULT_MAX_SHARD_SIZE = "1000GB"


def resolve_dtype(precision: str, sample_dtype: torch.dtype) -> tuple[str, torch.dtype]:
    normalized = precision.lower()
    if normalized in {"auto", "same", "source"}:
        if sample_dtype in {torch.bfloat16, torch.float16, torch.float32, torch.float64}:
            return str(sample_dtype).split(".")[-1], sample_dtype
        return "float32", torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return "bfloat16", torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return "float16", torch.float16
    if normalized in {"fp32", "float32"}:
        return "float32", torch.float32
    if normalized in {"fp64", "double", "float64"}:
        return "float64", torch.float64
    raise ValueError(f"Unsupported precision '{precision}'.")


def infer_max_position_embeddings(rwkv7: str, override: int | None = None) -> int:
    if override is not None:
        return override
    match = re.search(r"ctx(\d+)", Path(rwkv7).stem)
    if match:
        return int(match.group(1))
    return RWKV7Config().max_position_embeddings


def build_config(
    weights: dict[str, torch.Tensor],
    *,
    precision: str,
    max_position_embeddings: int,
) -> RWKV7Config:
    hidden_size = weights["blocks.0.ffn.key.weight"].shape[1]
    intermediate_size = weights["blocks.0.ffn.key.weight"].shape[0]
    num_hidden_layers = 0
    while f"blocks.{num_hidden_layers}.ffn.key.weight" in weights:
        num_hidden_layers += 1

    num_heads, head_dim = weights["blocks.0.att.r_k"].shape
    decay_low_rank_dim = weights["blocks.0.att.w1"].shape[1]
    gate_low_rank_dim = weights["blocks.0.att.g1"].shape[1]
    a_low_rank_dim = weights["blocks.0.att.a1"].shape[1]
    v_low_rank_dim = (
        weights["blocks.1.att.v1"].shape[1]
        if "blocks.1.att.v1" in weights
        else weights.get("blocks.0.att.v1", torch.empty(hidden_size, 32)).shape[1]
    )

    config = RWKV7Config(
        vocab_size=weights["emb.weight"].shape[0],
        hidden_size=hidden_size,
        hidden_ratio=intermediate_size // hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        head_dim=head_dim,
        num_heads=num_heads,
        value_dim=[hidden_size] * num_hidden_layers,
        decay_low_rank_dim=decay_low_rank_dim,
        gate_low_rank_dim=gate_low_rank_dim,
        a_low_rank_dim=a_low_rank_dim,
        v_low_rank_dim=v_low_rank_dim,
        max_position_embeddings=max_position_embeddings,
        bos_token_id=IM_START_TOKEN_ID,
        eos_token_id=IM_END_TOKEN_ID,
        pad_token_id=IM_END_TOKEN_ID,
        fuse_cross_entropy=False,
        fuse_linear_cross_entropy=False,
    )
    config.dtype = precision
    config.auto_map = {
        "AutoConfig": "configuration_rwkv7.RWKV7Config",
        "AutoModel": "modeling_rwkv7.RWKV7Model",
        "AutoModelForCausalLM": "modeling_rwkv7.RWKV7ForCausalLM",
    }
    return config


def translate_into_hf(
    name: str,
    *,
    num_hidden_layers: int,
) -> tuple[str, bool]:
    unused_names = {"blocks.0.att.v0", "blocks.0.att.v1", "blocks.0.att.v2"}
    if name in unused_names:
        return "", False

    emb_head = {
        "emb.weight": "model.embeddings.weight",
        "ln_out.weight": "model.norm.weight",
        "ln_out.bias": "model.norm.bias",
        "head.weight": "lm_head.weight",
    }
    proj = {
        "receptance": "r_proj",
        "key": "k_proj",
        "value": "v_proj",
        "ln_x": "g_norm",
        "output": "o_proj",
    }

    if name in emb_head:
        return emb_head[name], False

    name_parts = name.split(".")
    if name_parts[0] != "blocks":
        raise KeyError(f"Unexpected checkpoint key '{name}'.")

    layer_idx = int(name_parts[1])
    if layer_idx not in range(num_hidden_layers):
        raise KeyError(f"Unexpected layer index in '{name}'.")

    name_parts[0] = "model.layers"
    name_parts[2] = {
        "att": "attn",
        "ffn": "ffn",
        "ln0": "pre_norm",
        "ln1": "attn_norm",
        "ln2": "ffn_norm",
    }[name_parts[2]]

    transposed = False
    if re.fullmatch(r"[wvag][012]", name_parts[3]):
        typ, num = name_parts[3]
        name_parts[3] = f"{typ}_lora.lora." + {
            "0": "2.bias",
            "1": "0.weight",
            "2": "2.weight",
        }[num]
        transposed = num in {"1", "2"}
    elif name_parts[2] == "attn" and name_parts[3] in proj:
        name_parts[3] = proj[name_parts[3]]

    return ".".join(name_parts), transposed


def build_converted_state_dict(
    weights: dict[str, torch.Tensor],
    model: RWKV7ForCausalLM,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    model_dict = model.state_dict()
    pending_names = set(model_dict)
    converted: dict[str, torch.Tensor] = {}
    possible_absent_weights = {
        "model.layers.0.pre_norm.weight",
        "model.layers.0.pre_norm.bias",
    }

    for source_name, source_weight in weights.items():
        hf_name, transposed = translate_into_hf(
            source_name,
            num_hidden_layers=model.config.num_hidden_layers,
        )
        if not hf_name:
            continue

        weight = source_weight.detach()
        if transposed:
            weight = weight.t()

        target_tensor = model_dict[hf_name]
        if (
            weight.ndim == 3
            and weight.shape[:2] == (1, 1)
            and target_tensor.ndim == 1
            and weight.shape[-1] == target_tensor.shape[0]
        ):
            weight = weight.squeeze(0).squeeze(0)
        expected_shape = target_tensor.shape
        if weight.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {source_name} -> {hf_name}: "
                f"checkpoint={tuple(weight.shape)} expected={tuple(expected_shape)}"
            )

        converted[hf_name] = weight.to(dtype=dtype).contiguous()
        pending_names.discard(hf_name)

    missing_required = sorted(pending_names - possible_absent_weights)
    if missing_required:
        raise KeyError(f"Missing required parameters after conversion: {missing_required}")

    return converted


def build_model(config: RWKV7Config) -> RWKV7ForCausalLM:
    with torch.device("meta"):
        model = RWKV7ForCausalLM(config)
    return model


def convert(
    rwkv7: str,
    output: str,
    precision: str = "auto",
    max_position_embeddings: int | None = None,
):
    output = os.path.realpath(output)
    weights = torch.load(rwkv7, weights_only=True, map_location="cpu")
    text_weights = {}
    for name, tensor in weights.items():
        if name.startswith(("proj.", "encoder.", "vision.", "visual.")):
            continue
        if name.startswith("llm."):
            name = name.replace("llm.", "", 1)
        text_weights[name] = tensor

    if "emb.weight" not in text_weights:
        raise KeyError("Expected a text-only RWKV checkpoint with 'emb.weight'.")

    precision_name, dtype = resolve_dtype(precision, next(iter(text_weights.values())).dtype)
    config = build_config(
        text_weights,
        precision=precision_name,
        max_position_embeddings=infer_max_position_embeddings(rwkv7, max_position_embeddings),
    )
    print(f"Creating text-only RWKV7 HF model with config:\n{config}")

    RWKV7Config.register_for_auto_class()
    RWKV7ForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    model = build_model(config)
    converted_state = build_converted_state_dict(text_weights, model, dtype)
    missing, unexpected = model.load_state_dict(converted_state, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"Unexpected load_state_dict result: missing={missing}, unexpected={unexpected}")

    os.makedirs(output, exist_ok=True)
    model.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size=DEFAULT_MAX_SHARD_SIZE,
    )

    tokenizer = RwkvTokenizer(
        vocab_file=str(Path(__file__).with_name("rwkv_vocab_v20230424.txt")),
        bos_token=IM_START_TOKEN,
        eos_token=IM_END_TOKEN,
        pad_token=IM_END_TOKEN,
        unk_token=IM_START_TOKEN,
    )
    tokenizer.register_for_auto_class()
    tokenizer.save_pretrained(output)

    print(f"Saved text-only HF checkpoint to {output}")
    print("Verifying AutoConfig / AutoTokenizer / AutoModelForCausalLM loading...")
    loaded_config = AutoConfig.from_pretrained(output, trust_remote_code=True)
    loaded_tokenizer = AutoTokenizer.from_pretrained(output, trust_remote_code=True)
    loaded_model = AutoModelForCausalLM.from_pretrained(
        output,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    del loaded_config, loaded_model
    sample_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
    ]
    rendered_prompt = loaded_tokenizer.apply_chat_template(
        sample_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    sample_batch = loaded_tokenizer.apply_chat_template(
        sample_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(sample_batch, "keys"):
        prompt_shape = tuple(sample_batch["input_ids"].shape)
    else:
        prompt_shape = tuple(sample_batch.shape)
    print("Rendered chat_template preview:")
    print(rendered_prompt)
    print(f"Tokenizer verification prompt shape: {prompt_shape}")
    print(f"Export completed successfully: {output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert RWKV7')
    parser.add_argument('--rwkv7', type=str, help='Path to the input model')
    parser.add_argument('--output', type=str, help='Directory to save model')
    parser.add_argument('--precision', type=str, default='bfloat16')
    parser.add_argument('--max-position-embeddings', type=int, default=8192)
    args = parser.parse_args()
    convert(
        args.rwkv7,
        args.output,
        precision=args.precision,
        max_position_embeddings=args.max_position_embeddings,
    )
