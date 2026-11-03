# Modified in this distribution: V100 mixed-precision training. See NOTICE.
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer

from models.breeze import BreezeForConditionalGeneration
from models.breeze_config import BreezeConfig


def configure_eager_attention(config):
    """Propagate eager attention through object and dictionary sub-configs."""

    config._attn_implementation = "eager"
    config.use_cache = False
    for nested_name in (
        "backbone_config",
        "depth_decoder_config",
        "text_encoder_config",
    ):
        nested = getattr(config, nested_name, None)
        if nested is None:
            continue
        if isinstance(nested, dict):
            nested["_attn_implementation"] = "eager"
            nested["preferred_attn_implementation"] = "eager"
            nested["use_cache"] = False
            continue
        nested._attn_implementation = "eager"
        if hasattr(nested, "preferred_attn_implementation"):
            nested.preferred_attn_implementation = "eager"
        if hasattr(nested, "use_cache"):
            nested.use_cache = False
    return config


def load_eager_config(model_root: str | Path) -> BreezeConfig:
    """Load Breeze while propagating eager attention through every nested model."""

    config = BreezeConfig.from_pretrained(model_root)
    configure_eager_attention(config)
    return config


def load_training_model(
    model_root: str | Path,
    *,
    device: str,
) -> BreezeForConditionalGeneration:
    config = load_eager_config(model_root)
    dtype = torch.float32
    if device.startswith("cuda"):
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported(including_emulation=False)
            else torch.float16
        )
    model = BreezeForConditionalGeneration.from_pretrained(
        model_root,
        config=config,
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.to(device=device, dtype=dtype)
    if dtype == torch.float16 and model.text_encoder is not None:
        # This checkpoint's T5Gemma2 final norm exceeds FP16's finite range.
        model.text_encoder.float()
        model.text_encoder_proj.float()
    text_dtype = (
        next(model.text_encoder.parameters()).dtype
        if model.text_encoder is not None else None
    )
    print(f"Training model: {device}, synthesis={dtype}, text_encoder={text_dtype}", flush=True)
    model.config.use_cache = False
    return model


def load_tokenizer(model_root: str | Path):
    # Transformers 4.57.3 treats any local 4.57.3 model config as a possible
    # Mistral tokenizer. Breeze uses GemmaTokenizerFast with one whitespace
    # Split pre-tokenizer, so the Mistral sequence-index patch is inapplicable.
    return AutoTokenizer.from_pretrained(model_root, fix_mistral_regex=False)
