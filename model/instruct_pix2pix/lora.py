"""LoRA injection for InstructPix2Pix UNet fine-tuning.

Adds low-rank adapters (rank 16 by default) to all attention projection layers
(q, k, v, out_proj) in the UNet. Only LoRA parameters are trained; base weights
are frozen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 16.0          # scaling factor: alpha / rank
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"]
    )

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @classmethod
    def from_dict(cls, d: dict) -> "LoRAConfig":
        return cls(
            rank=d.get("rank", 16),
            alpha=d.get("alpha", 16.0),
            dropout=d.get("dropout", 0.0),
            target_modules=d.get("target_modules", cls().target_modules),
        )


class LoRALinear(nn.Module):
    """Wraps a frozen Linear layer with a low-rank additive adapter."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = linear
        self.rank = rank
        self.scale = alpha / rank

        in_features = linear.in_features
        out_features = linear.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        device = linear.weight.device
        self.lora_A = self.lora_A.to(device=device)  # stay fp32 for GradScaler
        self.lora_B = self.lora_B.to(device=device)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # LoRA weights are fp32; cast input up, cast result back to match base output
        lora_out = self.lora_B(self.lora_A(self.dropout(x.to(self.lora_A.weight.dtype)))) * self.scale
        return base_out + lora_out.to(base_out.dtype)

    def merge_into_base(self) -> nn.Linear:
        """Return a new Linear with the LoRA delta merged in (for inference export)."""
        merged = nn.Linear(
            self.base.in_features, self.base.out_features,
            bias=self.base.bias is not None,
        )
        merged.weight = nn.Parameter(
            self.base.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scale
        )
        if self.base.bias is not None:
            merged.bias = nn.Parameter(self.base.bias.clone())
        return merged


def inject_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Replace target Linear layers in-place with LoRALinear wrappers.

    Returns the model with LoRA injected and all non-LoRA parameters frozen.
    """
    _replace_layers(model, config)
    # Freeze everything that is NOT a LoRA parameter
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad = False
    return model


def _replace_layers(module: nn.Module, config: LoRAConfig, prefix: str = "") -> None:
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and _is_target(name, config.target_modules):
            setattr(module, name, LoRALinear(child, config.rank, config.alpha, config.dropout))
        else:
            _replace_layers(child, config, full_name)


def _is_target(name: str, target_modules: list[str]) -> bool:
    return any(name == t or name.endswith(f".{t}") or t in name for t in target_modules)


def count_trainable_params(model: nn.Module) -> dict[str, int]:
    """Return counts of trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "frozen": total - trainable}
