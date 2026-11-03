from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from training.model_loading import configure_eager_attention, load_training_model


def test_configure_eager_attention_handles_mixed_nested_config_shapes():
    backbone = {"model_type": "qwen3"}
    depth_decoder = SimpleNamespace(use_cache=True)
    text_encoder = SimpleNamespace(preferred_attn_implementation="flash_attention_2")
    config = SimpleNamespace(
        use_cache=True,
        backbone_config=backbone,
        depth_decoder_config=depth_decoder,
        text_encoder_config=text_encoder,
    )

    configure_eager_attention(config)

    assert config._attn_implementation == "eager"
    assert config.use_cache is False
    assert backbone["_attn_implementation"] == "eager"
    assert backbone["preferred_attn_implementation"] == "eager"
    assert backbone["use_cache"] is False
    assert depth_decoder._attn_implementation == "eager"
    assert depth_decoder.use_cache is False
    assert text_encoder._attn_implementation == "eager"
    assert text_encoder.preferred_attn_implementation == "eager"


@pytest.mark.parametrize("device,bf16,expected", [
    ("cpu", False, torch.float32),
    ("cuda:0", False, torch.float16),
    ("cuda:0", True, torch.bfloat16),
])
def test_training_precision_policy_without_cuda(tmp_path, device, bf16, expected):
    model = MagicMock()
    model.text_encoder.parameters.return_value = iter([torch.nn.Parameter(torch.ones(1))])
    with (
        patch("training.model_loading.load_eager_config"),
        patch("training.model_loading.BreezeForConditionalGeneration.from_pretrained", return_value=model) as loader,
        patch("training.model_loading.torch.cuda.is_bf16_supported", return_value=bf16),
    ):
        assert load_training_model(tmp_path, device=device) is model
    assert loader.call_args.kwargs["dtype"] == expected
    model.to.assert_called_once_with(device=device, dtype=expected)
    if expected == torch.float16:
        model.text_encoder.float.assert_called_once()
        model.text_encoder_proj.float.assert_called_once()
    else:
        model.text_encoder.float.assert_not_called()
    assert model.config.use_cache is False
