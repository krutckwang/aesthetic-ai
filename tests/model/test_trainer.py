"""Tests for InstructPix2PixTrainer and TrainerConfig."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import torch
import torch.nn as nn

from model.training.trainer import TrainerConfig, InstructPix2PixTrainer, _infinite


# ── TrainerConfig ─────────────────────────────────────────────────────────────

def test_default_num_steps():
    assert TrainerConfig().num_steps == 15_000


def test_default_mixed_precision():
    assert TrainerConfig().mixed_precision == "fp16"


def test_from_dict_parses_known_keys():
    cfg = TrainerConfig.from_dict({
        "num_steps": 500,
        "learning_rate": 5e-5,
        "mixed_precision": "no",
    })
    assert cfg.num_steps == 500
    assert cfg.learning_rate == 5e-5
    assert cfg.mixed_precision == "no"


def test_from_dict_ignores_unknown_keys():
    # Should not raise
    cfg = TrainerConfig.from_dict({"num_steps": 100, "unknown_key": "ignored"})
    assert cfg.num_steps == 100


# ── _infinite ─────────────────────────────────────────────────────────────────

def test_infinite_cycles():
    data = [1, 2, 3]
    gen = _infinite(data)
    results = [next(gen) for _ in range(9)]
    assert results == [1, 2, 3, 1, 2, 3, 1, 2, 3]


# ── InstructPix2PixTrainer setup ──────────────────────────────────────────────

def _make_trainer(tmp_path, num_steps=2):
    """Build a trainer with fully mocked components."""
    unet = _tiny_unet()
    vae = MagicMock()
    vae.config.scaling_factor = 0.18215
    latent = torch.zeros(1, 4, 8, 8)
    vae.encode.return_value.latent_dist.sample.return_value = latent
    vae.parameters.return_value = iter([])  # nothing to freeze

    text_encoder = MagicMock()
    text_encoder.config.hidden_size = 64
    text_encoder.return_value = [torch.zeros(1, 1, 64)]
    text_encoder.parameters.return_value = iter([])  # nothing to freeze

    tokenizer = MagicMock()

    scheduler = MagicMock()
    scheduler.config.num_train_timesteps = 1000
    scheduler.add_noise.return_value = torch.zeros(1, 4, 8, 8)

    batch = {
        "pixel_values": torch.zeros(1, 3, 64, 64),
        "edited_pixel_values": torch.zeros(1, 3, 64, 64),
    }
    dl = [batch]

    cfg = TrainerConfig(
        output_dir=str(tmp_path / "lora"),
        num_steps=num_steps,
        mixed_precision="no",
        save_every_n_steps=1,
    )
    trainer = InstructPix2PixTrainer(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        noise_scheduler=scheduler,
        tokenizer=tokenizer,
        train_dataloader=dl,
        val_dataloader=None,
        config=cfg,
        device="cpu",
    )
    return trainer


class _TinyUNet(nn.Module):
    """Minimal UNet that participates in the gradient graph (input shape: B×8×H×W → B×4×H×W)."""

    def __init__(self):
        super().__init__()
        # 8-channel latent concat → 4-channel noise pred (flattened over H×W=8×8=64)
        self.linear = nn.Linear(8 * 8 * 8, 4 * 8 * 8, bias=False)

    def forward(self, x, t, encoder_hidden_states):
        batch = x.shape[0]
        h, w = x.shape[-2], x.shape[-1]
        flat = x.reshape(batch, -1).float()
        out = self.linear(flat).reshape(batch, 4, h, w)
        result = MagicMock()
        result.sample = out
        return result


def _tiny_unet() -> nn.Module:
    return _TinyUNet()


def test_setup_creates_optimizer(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.setup()
    assert trainer._optimizer is not None


def test_setup_no_scaler_for_fp32(tmp_path):
    trainer = _make_trainer(tmp_path)
    trainer.setup()
    assert trainer._scaler is None  # mixed_precision="no"


def test_train_saves_checkpoint(tmp_path):
    trainer = _make_trainer(tmp_path, num_steps=2)
    trainer.train()
    # final checkpoint must exist
    final = tmp_path / "lora" / "final" / "lora_weights.pt"
    assert final.exists()


def test_train_returns_stats(tmp_path):
    trainer = _make_trainer(tmp_path, num_steps=2)
    stats = trainer.train()
    assert "avg_loss" in stats
    assert "steps" in stats
    assert stats["steps"] == 2


def test_checkpoint_contains_lora_keys(tmp_path):
    """Checkpoint must only save LoRA-named parameters."""
    from model.instruct_pix2pix.lora import LoRAConfig, inject_lora

    trainer = _make_trainer(tmp_path, num_steps=1)
    # Inject LoRA into the tiny unet so it has lora_A/lora_B keys
    cfg = LoRAConfig(rank=2, alpha=2.0, target_modules=["linear"])
    inject_lora(trainer.unet, cfg)
    trainer.train()
    ckpt = torch.load(tmp_path / "lora" / "final" / "lora_weights.pt")
    assert all("lora_A" in k or "lora_B" in k for k in ckpt.keys())
