"""Tests for LoRA injection."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Mock torch before any model import
_torch = MagicMock()
_torch.nn = MagicMock()

import torch  # real torch may not be installed; try real first, fall back to mock
import torch.nn as nn

import pytest
import math

from model.instruct_pix2pix.lora import (
    LoRAConfig, LoRALinear, inject_lora, count_trainable_params, _is_target,
)


# ── LoRAConfig ────────────────────────────────────────────────────────────────

def test_default_rank_is_16():
    assert LoRAConfig().rank == 16


def test_scale_equals_alpha_over_rank():
    cfg = LoRAConfig(rank=8, alpha=16.0)
    assert cfg.scale == 2.0


def test_from_dict_parses_rank():
    cfg = LoRAConfig.from_dict({"rank": 4, "alpha": 8.0})
    assert cfg.rank == 4
    assert cfg.alpha == 8.0


def test_from_dict_uses_defaults_for_missing_keys():
    cfg = LoRAConfig.from_dict({})
    assert cfg.rank == 16
    assert cfg.dropout == 0.0


# ── _is_target ────────────────────────────────────────────────────────────────

def test_is_target_exact_match():
    assert _is_target("to_q", ["to_q", "to_k"])


def test_is_target_substring_match():
    assert _is_target("attention.to_q", ["to_q"])


def test_is_not_target():
    assert not _is_target("conv_in", ["to_q", "to_k", "to_v"])


# ── LoRALinear ────────────────────────────────────────────────────────────────

def test_lora_linear_output_shape():
    base = nn.Linear(64, 32, bias=False)
    lora = LoRALinear(base, rank=4, alpha=4.0)
    x = torch.randn(2, 64)
    out = lora(x)
    assert out.shape == (2, 32)


def test_lora_linear_base_frozen():
    base = nn.Linear(32, 16, bias=False)
    lora = LoRALinear(base, rank=4, alpha=4.0)
    assert not lora.base.weight.requires_grad


def test_lora_a_b_trainable():
    base = nn.Linear(32, 16, bias=False)
    lora = LoRALinear(base, rank=4, alpha=4.0)
    assert lora.lora_A.weight.requires_grad
    assert lora.lora_B.weight.requires_grad


def test_lora_b_init_zeros():
    base = nn.Linear(64, 32, bias=False)
    lora = LoRALinear(base, rank=8, alpha=8.0)
    assert lora.lora_B.weight.abs().max().item() == 0.0


def test_lora_linear_with_bias():
    base = nn.Linear(16, 8, bias=True)
    lora = LoRALinear(base, rank=2, alpha=2.0)
    x = torch.randn(3, 16)
    out = lora(x)
    assert out.shape == (3, 8)


def test_merge_into_base_output_unchanged():
    """Merged linear should produce same output as LoRALinear (within fp32 tolerance)."""
    base = nn.Linear(32, 16, bias=False)
    lora = LoRALinear(base, rank=4, alpha=4.0)
    # Assign non-zero B to make delta visible
    nn.init.normal_(lora.lora_B.weight)
    nn.init.normal_(lora.lora_A.weight)

    x = torch.randn(2, 32)
    with torch.no_grad():
        expected = lora(x)
        merged = lora.merge_into_base()
        actual = merged(x)

    assert torch.allclose(expected, actual, atol=1e-5)


# ── inject_lora ───────────────────────────────────────────────────────────────

class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(16, 16, bias=False)
        self.to_k = nn.Linear(16, 16, bias=False)
        self.other = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.to_q(x) + self.to_k(x) + self.other(x)


def test_inject_lora_replaces_target_layers():
    model = _TinyModel()
    cfg = LoRAConfig(rank=4, alpha=4.0, target_modules=["to_q", "to_k"])
    inject_lora(model, cfg)
    assert isinstance(model.to_q, LoRALinear)
    assert isinstance(model.to_k, LoRALinear)


def test_inject_lora_preserves_non_target_layers():
    model = _TinyModel()
    cfg = LoRAConfig(rank=4, alpha=4.0, target_modules=["to_q", "to_k"])
    inject_lora(model, cfg)
    assert isinstance(model.other, nn.Linear)
    assert not isinstance(model.other, LoRALinear)


def test_inject_lora_freezes_non_lora():
    model = _TinyModel()
    cfg = LoRAConfig(rank=4, alpha=4.0, target_modules=["to_q"])
    inject_lora(model, cfg)
    for name, p in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            assert not p.requires_grad, f"{name} should be frozen"


# ── count_trainable_params ────────────────────────────────────────────────────

def test_count_trainable_params_structure():
    model = _TinyModel()
    cfg = LoRAConfig(rank=4, alpha=4.0, target_modules=["to_q", "to_k"])
    inject_lora(model, cfg)
    counts = count_trainable_params(model)
    assert "trainable" in counts and "total" in counts and "frozen" in counts
    assert counts["trainable"] + counts["frozen"] == counts["total"]


def test_lora_reduces_trainable_fraction():
    model = _TinyModel()
    total_before = sum(p.numel() for p in model.parameters())
    cfg = LoRAConfig(rank=4, alpha=4.0, target_modules=["to_q", "to_k"])
    inject_lora(model, cfg)
    counts = count_trainable_params(model)
    # LoRA params should be much fewer than total
    assert counts["trainable"] < counts["total"]
    assert counts["trainable"] < total_before * 0.5
