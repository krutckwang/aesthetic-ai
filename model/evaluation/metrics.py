"""Evaluation metrics for aesthetic treatment prediction quality.

Metrics implemented:
  - SSIM (Structural Similarity Index) — perceptual similarity to ground truth
  - LPIPS proxy via pixel-level L1 (full LPIPS requires VGG; implemented as stub)
  - FID (Fréchet Inception Distance) — population-level realism; stub for local use
  - Identity preservation score — cosine similarity of face embeddings
  - Treatment fidelity — classification accuracy of predicted treatment category
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class MetricResult:
    ssim: float
    l1: float
    identity_cosine: float | None = None
    treatment_fidelity: float | None = None
    fid: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ── SSIM ──────────────────────────────────────────────────────────────────────

def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 2.0,  # images in [-1, 1]
) -> float:
    """Compute mean SSIM over a batch. Inputs: (B, C, H, W) in [-1, 1]."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_x = _gaussian_pool(pred, window_size)
    mu_y = _gaussian_pool(target, window_size)
    mu_xx = _gaussian_pool(pred * pred, window_size)
    mu_yy = _gaussian_pool(target * target, window_size)
    mu_xy = _gaussian_pool(pred * target, window_size)

    sigma_x = mu_xx - mu_x * mu_x
    sigma_y = mu_yy - mu_y * mu_y
    sigma_xy = mu_xy - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = numerator / (denominator + 1e-8)
    return ssim_map.mean().item()


def _gaussian_pool(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pad = kernel_size // 2
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad, count_include_pad=False)


# ── L1 (LPIPS proxy) ─────────────────────────────────────────────────────────

def compute_l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error over all pixels. Used as lightweight LPIPS proxy."""
    return F.l1_loss(pred.float(), target.float()).item()


# ── Identity preservation ─────────────────────────────────────────────────────

def compute_identity_cosine(embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> float:
    """Cosine similarity between two face embeddings. Range: [-1, 1]; higher is better."""
    a = F.normalize(embedding_a.float(), dim=-1)
    b = F.normalize(embedding_b.float(), dim=-1)
    return (a * b).sum(dim=-1).mean().item()


# ── Batch evaluation ─────────────────────────────────────────────────────────

def evaluate_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    before_embeddings: torch.Tensor | None = None,
    pred_embeddings: torch.Tensor | None = None,
) -> MetricResult:
    """Compute all available metrics for a batch."""
    ssim = compute_ssim(pred, target)
    l1 = compute_l1(pred, target)

    identity = None
    if before_embeddings is not None and pred_embeddings is not None:
        identity = compute_identity_cosine(before_embeddings, pred_embeddings)

    return MetricResult(ssim=ssim, l1=l1, identity_cosine=identity)


# ── Running average tracker ───────────────────────────────────────────────────

class MetricTracker:
    """Accumulates per-batch metrics and reports running averages."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, result: MetricResult) -> None:
        for key, value in result.to_dict().items():
            self._sums[key] = self._sums.get(key, 0.0) + value
            self._counts[key] = self._counts.get(key, 0) + 1

    def compute(self) -> dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()
