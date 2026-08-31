"""Tests for CycleGAN model and trainer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from model.cyclegan.model import (
    ResidualBlock, ResNetGenerator, PatchGANDiscriminator, CycleGAN,
)
from model.cyclegan.trainer import CycleGANTrainerConfig, CycleGANTrainer


# ── ResidualBlock ─────────────────────────────────────────────────────────────

def test_residual_block_output_shape():
    block = ResidualBlock(16)
    x = torch.randn(1, 16, 32, 32)
    out = block(x)
    assert out.shape == x.shape


def test_residual_block_is_skip_connection():
    # Sequential: 0=Pad, 1=Conv2d, 2=IN, 3=ReLU, 4=Pad, 5=Conv2d, 6=IN
    block = ResidualBlock(8)
    x = torch.zeros(1, 8, 16, 16)
    with torch.no_grad():
        nn.init.zeros_(block.block[1].weight)
        block.block[1].bias.data.zero_()
        nn.init.zeros_(block.block[5].weight)
        block.block[5].bias.data.zero_()
    out = block(x)
    assert torch.allclose(out, x, atol=1e-6)


# ── ResNetGenerator ───────────────────────────────────────────────────────────

def test_generator_output_shape():
    gen = ResNetGenerator(n_residual_blocks=2)  # fewer blocks for test speed
    x = torch.randn(1, 3, 64, 64)
    out = gen(x)
    assert out.shape == (1, 3, 64, 64)


def test_generator_output_range():
    gen = ResNetGenerator(n_residual_blocks=2)
    x = torch.randn(1, 3, 32, 32)
    out = gen(x)
    assert out.min() >= -1.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_generator_batch_invariance():
    gen = ResNetGenerator(n_residual_blocks=1)
    x1 = torch.randn(1, 3, 32, 32)
    x2 = torch.randn(3, 3, 32, 32)
    out1 = gen(x1)
    out2 = gen(x2)
    assert out1.shape == (1, 3, 32, 32)
    assert out2.shape == (3, 3, 32, 32)


# ── PatchGANDiscriminator ─────────────────────────────────────────────────────

def test_discriminator_output_is_grid():
    disc = PatchGANDiscriminator()
    x = torch.randn(1, 3, 256, 256)
    out = disc(x)
    # Should be a spatial grid, not a scalar
    assert out.ndim == 4
    assert out.shape[1] == 1  # single channel logit map


def test_discriminator_small_input():
    disc = PatchGANDiscriminator(base_filters=8, n_layers=2)
    x = torch.randn(2, 3, 64, 64)
    out = disc(x)
    assert out.shape[0] == 2
    assert out.shape[1] == 1


# ── CycleGAN ──────────────────────────────────────────────────────────────────

def _tiny_cyclegan() -> CycleGAN:
    return CycleGAN(base_filters=8, n_residual_blocks=1)


def test_cyclegan_has_both_generators():
    model = _tiny_cyclegan()
    assert hasattr(model, "G_AB")
    assert hasattr(model, "G_BA")


def test_generator_loss_keys():
    model = _tiny_cyclegan()
    a = torch.randn(1, 3, 32, 32)
    b = torch.randn(1, 3, 32, 32)
    losses = model.compute_generator_loss(a, b)
    for key in ("total", "adv_AB", "adv_BA", "cycle_A", "cycle_B", "idt_A", "idt_B"):
        assert key in losses


def test_discriminator_loss_keys():
    model = _tiny_cyclegan()
    a = torch.randn(1, 3, 32, 32)
    b = torch.randn(1, 3, 32, 32)
    losses = model.compute_discriminator_loss(a, b)
    assert "total" in losses
    assert "D_A" in losses
    assert "D_B" in losses


def test_generator_loss_backward():
    model = _tiny_cyclegan()
    a = torch.randn(1, 3, 32, 32, requires_grad=False)
    b = torch.randn(1, 3, 32, 32, requires_grad=False)
    losses = model.compute_generator_loss(a, b)
    losses["total"].backward()   # should not raise


def test_cycle_loss_greater_than_zero():
    model = _tiny_cyclegan()
    a = torch.randn(1, 3, 32, 32)
    b = torch.randn(1, 3, 32, 32) + 10.0  # very different domain
    losses = model.compute_generator_loss(a, b)
    assert losses["cycle_A"].item() > 0


# ── CycleGANTrainerConfig ─────────────────────────────────────────────────────

def test_trainer_config_defaults():
    cfg = CycleGANTrainerConfig()
    assert cfg.lambda_cycle == 10.0
    assert cfg.lambda_identity == 0.5


def test_trainer_config_from_dict():
    cfg = CycleGANTrainerConfig.from_dict({"num_steps": 100, "lambda_cycle": 5.0})
    assert cfg.num_steps == 100
    assert cfg.lambda_cycle == 5.0


# ── CycleGANTrainer ───────────────────────────────────────────────────────────

def test_cyclegan_trainer_saves_checkpoint(tmp_path):
    model = _tiny_cyclegan()
    batch = {
        "pixel_values": torch.randn(1, 3, 32, 32),
        "edited_pixel_values": torch.randn(1, 3, 32, 32),
    }
    cfg = CycleGANTrainerConfig(
        output_dir=str(tmp_path / "cyclegan"),
        num_steps=2,
        save_every_n_steps=5,
    )
    trainer = CycleGANTrainer(model, [batch], cfg, device="cpu")
    stats = trainer.train()
    assert (tmp_path / "cyclegan" / "final" / "cyclegan.pt").exists()
    assert stats["steps"] == 2


def test_cyclegan_trainer_returns_stats(tmp_path):
    model = _tiny_cyclegan()
    batch = {
        "pixel_values": torch.randn(1, 3, 32, 32),
        "edited_pixel_values": torch.randn(1, 3, 32, 32),
    }
    cfg = CycleGANTrainerConfig(
        output_dir=str(tmp_path / "cyclegan"),
        num_steps=1,
        save_every_n_steps=100,
    )
    trainer = CycleGANTrainer(model, [batch], cfg, device="cpu")
    stats = trainer.train()
    assert "avg_g_loss" in stats
    assert "avg_d_loss" in stats
