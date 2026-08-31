"""Tests for evaluation metrics and evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from model.evaluation.metrics import (
    compute_ssim, compute_l1, compute_identity_cosine,
    evaluate_batch, MetricResult, MetricTracker,
)
from model.evaluation.evaluator import EvaluationReport, Evaluator


# ── compute_l1 ────────────────────────────────────────────────────────────────

def test_l1_identical_images_is_zero():
    x = torch.rand(1, 3, 32, 32)
    assert compute_l1(x, x) == pytest.approx(0.0, abs=1e-6)


def test_l1_opposite_images():
    x = torch.ones(1, 3, 8, 8)
    y = -torch.ones(1, 3, 8, 8)
    assert compute_l1(x, y) == pytest.approx(2.0, abs=1e-4)


def test_l1_is_positive():
    x = torch.rand(2, 3, 16, 16)
    y = torch.rand(2, 3, 16, 16)
    assert compute_l1(x, y) >= 0.0


# ── compute_ssim ──────────────────────────────────────────────────────────────

def test_ssim_identical_is_one():
    x = torch.rand(1, 3, 64, 64)
    ssim = compute_ssim(x, x)
    assert ssim == pytest.approx(1.0, abs=1e-3)


def test_ssim_range():
    x = torch.rand(2, 3, 64, 64) * 2 - 1  # [-1, 1]
    y = torch.rand(2, 3, 64, 64) * 2 - 1
    ssim = compute_ssim(x, y)
    assert -1.0 <= ssim <= 1.0


def test_ssim_decreases_with_noise():
    x = torch.rand(1, 3, 64, 64)
    noisy = x + 0.5 * torch.randn_like(x)
    assert compute_ssim(x, x) > compute_ssim(x, noisy)


# ── compute_identity_cosine ───────────────────────────────────────────────────

def test_identity_cosine_same_embedding_is_one():
    e = torch.randn(1, 512)
    assert compute_identity_cosine(e, e) == pytest.approx(1.0, abs=1e-5)


def test_identity_cosine_opposite_is_minus_one():
    e = torch.randn(1, 512)
    assert compute_identity_cosine(e, -e) == pytest.approx(-1.0, abs=1e-5)


def test_identity_cosine_range():
    a = torch.randn(4, 256)
    b = torch.randn(4, 256)
    val = compute_identity_cosine(a, b)
    assert -1.0 <= val <= 1.0


# ── evaluate_batch ────────────────────────────────────────────────────────────

def test_evaluate_batch_returns_metric_result():
    pred = torch.rand(2, 3, 32, 32)
    target = torch.rand(2, 3, 32, 32)
    result = evaluate_batch(pred, target)
    assert isinstance(result, MetricResult)


def test_evaluate_batch_with_embeddings():
    pred = torch.rand(1, 3, 32, 32)
    target = torch.rand(1, 3, 32, 32)
    e1 = torch.randn(1, 512)
    e2 = torch.randn(1, 512)
    result = evaluate_batch(pred, target, before_embeddings=e1, pred_embeddings=e2)
    assert result.identity_cosine is not None


def test_evaluate_batch_no_embeddings_has_none_identity():
    pred = torch.rand(1, 3, 32, 32)
    target = torch.rand(1, 3, 32, 32)
    result = evaluate_batch(pred, target)
    assert result.identity_cosine is None


# ── MetricTracker ─────────────────────────────────────────────────────────────

def test_tracker_averages_ssim():
    tracker = MetricTracker()
    tracker.update(MetricResult(ssim=0.8, l1=0.2))
    tracker.update(MetricResult(ssim=0.6, l1=0.4))
    out = tracker.compute()
    assert out["ssim"] == pytest.approx(0.7, abs=1e-6)
    assert out["l1"] == pytest.approx(0.3, abs=1e-6)


def test_tracker_reset():
    tracker = MetricTracker()
    tracker.update(MetricResult(ssim=0.9, l1=0.1))
    tracker.reset()
    assert tracker.compute() == {}


def test_tracker_to_dict_excludes_none():
    result = MetricResult(ssim=0.8, l1=0.1, identity_cosine=None)
    d = result.to_dict()
    assert "identity_cosine" not in d
    assert "ssim" in d


# ── EvaluationReport ──────────────────────────────────────────────────────────

def test_report_save_and_load(tmp_path):
    report = EvaluationReport(
        model_tag="lora_v1",
        num_samples=100,
        metrics={"ssim": 0.75, "l1": 0.12},
    )
    path = tmp_path / "report.json"
    report.save(path)
    loaded = json.loads(path.read_text())
    assert loaded["model_tag"] == "lora_v1"
    assert loaded["metrics"]["ssim"] == pytest.approx(0.75)


def test_report_to_dict_structure():
    report = EvaluationReport(model_tag="test", num_samples=10, metrics={"ssim": 0.5})
    d = report.to_dict()
    assert "model_tag" in d and "num_samples" in d and "metrics" in d


# ── Evaluator ─────────────────────────────────────────────────────────────────

class _IdentityModel:
    """Returns before image unchanged (worst-case baseline)."""
    def eval(self): pass


def test_evaluator_returns_report():
    model = _IdentityModel()
    evaluator = Evaluator(model, model_tag="identity_baseline")
    batch = {
        "pixel_values": torch.rand(2, 3, 32, 32),
        "edited_pixel_values": torch.rand(2, 3, 32, 32),
    }
    report = evaluator.evaluate([batch])
    assert isinstance(report, EvaluationReport)
    assert report.num_samples == 2


def test_evaluator_reports_ssim():
    model = _IdentityModel()
    evaluator = Evaluator(model, model_tag="identity")
    batch = {
        "pixel_values": torch.rand(1, 3, 32, 32),
        "edited_pixel_values": torch.rand(1, 3, 32, 32),
    }
    report = evaluator.evaluate([batch])
    assert "ssim" in report.metrics


def test_evaluator_per_treatment_breakdown():
    model = _IdentityModel()
    evaluator = Evaluator(model, model_tag="identity")
    batch = {
        "pixel_values": torch.rand(2, 3, 32, 32),
        "edited_pixel_values": torch.rand(2, 3, 32, 32),
        "treatment_category": ["botox", "lip_filler"],
    }
    report = evaluator.evaluate([batch])
    assert "botox" in report.per_treatment
    assert "lip_filler" in report.per_treatment
