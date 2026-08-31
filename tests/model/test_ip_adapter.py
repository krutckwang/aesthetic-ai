"""Tests for IP-Adapter config and ImageProjection module."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from model.ip_adapter.adapter import IPAdapterConfig, ImageProjection, IPAdapter


# ── IPAdapterConfig ───────────────────────────────────────────────────────────

def test_default_scale():
    assert IPAdapterConfig().scale == 0.8


def test_default_num_tokens():
    assert IPAdapterConfig().num_tokens == 4


def test_from_dict_parses_scale():
    cfg = IPAdapterConfig.from_dict({"scale": 0.5})
    assert cfg.scale == 0.5


def test_from_dict_defaults():
    cfg = IPAdapterConfig.from_dict({})
    assert cfg.scale == 0.8
    assert cfg.num_tokens == 4


# ── ImageProjection ───────────────────────────────────────────────────────────

def test_image_projection_output_shape():
    proj = ImageProjection(clip_dim=768, cross_attention_dim=768, num_tokens=4)
    embeds = torch.randn(2, 768)
    out = proj(embeds)
    assert out.shape == (2, 4, 768)


def test_image_projection_different_num_tokens():
    proj = ImageProjection(clip_dim=512, cross_attention_dim=512, num_tokens=16)
    embeds = torch.randn(3, 512)
    out = proj(embeds)
    assert out.shape == (3, 16, 512)


def test_image_projection_layer_norm_applied():
    proj = ImageProjection(clip_dim=64, cross_attention_dim=64, num_tokens=2)
    embeds = torch.randn(1, 64) * 100  # large scale
    out = proj(embeds)
    # After LayerNorm, values should be in a reasonable range
    assert out.abs().max().item() < 50.0


def test_image_projection_trainable():
    proj = ImageProjection()
    trainable = [p for p in proj.parameters() if p.requires_grad]
    assert len(trainable) > 0


# ── IPAdapter ─────────────────────────────────────────────────────────────────

def test_ip_adapter_set_scale():
    mock_pipe = MagicMock()
    adapter = IPAdapter(mock_pipe)
    adapter.set_scale(0.3)
    assert adapter.config.scale == 0.3


def test_ip_adapter_get_image_tokens_raises_without_weights():
    mock_pipe = MagicMock()
    adapter = IPAdapter(mock_pipe)
    with pytest.raises(RuntimeError, match="load_weights"):
        adapter.get_image_tokens(MagicMock())


def test_ip_adapter_load_weights(tmp_path):
    mock_pipe = MagicMock()
    adapter = IPAdapter(mock_pipe)
    proj = ImageProjection()
    weights_path = tmp_path / "proj.pt"
    torch.save(proj.state_dict(), str(weights_path))
    adapter.load_weights(weights_path)
    assert adapter._loaded is True
    assert adapter._projection is not None
