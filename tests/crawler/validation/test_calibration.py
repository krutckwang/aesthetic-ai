"""Tests for crawler/validation/calibration.py — threshold calibration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from crawler.validation.calibration import (
    Calibrator,
    CalibrationResult,
    CalibrationSample,
)
from crawler.validation.structural import StructuralResult
from crawler.validation.nlp_pairing import NLPPairingResult
from crawler.validation.face_similarity import FaceSimilarityResult


def _pos_sample(struct_conf: float = 0.90, nlp_score: float = 0.85) -> CalibrationSample:
    return CalibrationSample(
        is_positive=True,
        structural=StructuralResult(
            has_explicit_label=True,
            confidence=struct_conf,
            before_label_signals=["strong:alt_text:before"],
            after_label_signals=["strong:alt_text:after"],
        ),
        nlp=NLPPairingResult(
            pairing_score=nlp_score,
            detected_language="en",
            method="keyword",
            before_text_snippet="before",
            after_text_snippet="after",
        ),
        arcface=None,
    )


def _neg_sample(struct_conf: float = 0.20, nlp_score: float = 0.15) -> CalibrationSample:
    return CalibrationSample(
        is_positive=False,
        structural=StructuralResult(
            has_explicit_label=False,
            confidence=struct_conf,
            before_label_signals=[],
            after_label_signals=[],
        ),
        nlp=NLPPairingResult(
            pairing_score=nlp_score,
            detected_language="en",
            method="semantic",
            before_text_snippet="random",
            after_text_snippet="text",
        ),
        arcface=None,
    )


@pytest.fixture
def validation_config(tmp_path: Path) -> Path:
    config = {
        "calibrated": False,
        "layer1": {"min_confidence": 0.70},
        "layer2": {"min_pairing_score": 0.65},
        "layer3": {"min_cosine_similarity": 0.40},
    }
    path = tmp_path / "validation.yaml"
    path.write_text(yaml.dump(config))
    return path


@pytest.fixture
def mock_queue(tmp_path: Path):
    queue = MagicMock()
    queue.dequeue_batch.return_value = []
    return queue


class TestCalibrator:
    def test_precision_at_threshold(self):
        """precision_at should correctly compute TP / (TP + FP)."""
        samples = [_pos_sample(0.9), _pos_sample(0.85), _neg_sample(0.3), _neg_sample(0.1)]
        from crawler.storage.staging_queue import StagingQueue
        from pathlib import Path
        queue = MagicMock()
        cal = Calibrator(queue=queue, config_path=Path("/dev/null"), n_samples=4)

        prec = cal._precision_at(samples, "structural", 0.7)
        # TP = 2 (conf 0.9, 0.85 >= 0.7), FP = 1 (neg 0.3 >= 0.7? No. neg 0.1? No)
        assert prec == 1.0

    def test_recall_at_threshold(self):
        samples = [_pos_sample(0.9), _pos_sample(0.4), _neg_sample(0.1)]
        queue = MagicMock()
        cal = Calibrator(queue=queue, config_path=Path("/dev/null"), n_samples=3)

        recall = cal._recall_at(samples, "structural", 0.5)
        # TP = 1 (only 0.9 >= 0.5), FN = 1 (0.4 < 0.5)
        assert recall == 0.5

    def test_calibrate_writes_config(self, mock_queue, validation_config):
        """After calibration with clean positive/negative sets, config gets updated."""
        positives = [_pos_sample() for _ in range(20)]
        negatives = [_neg_sample() for _ in range(20)]
        all_samples = positives + negatives

        cal = Calibrator(
            queue=mock_queue,
            config_path=validation_config,
            n_samples=20,
        )

        # Bypass the queue-fetching logic and inject samples directly
        with patch.object(cal, "_collect_positive_samples", return_value=positives), \
             patch.object(cal, "_build_negative_samples", return_value=negatives):
            result = cal.run()

        # Config should now have calibrated=True
        with open(validation_config) as f:
            written = yaml.safe_load(f)

        assert written["calibrated"] is True
        assert "layer1" in written
        assert "layer2" in written
        assert result.n_positives == 20
        assert result.n_negatives == 20

    def test_calibration_result_fields_populated(self, mock_queue, validation_config):
        positives = [_pos_sample() for _ in range(10)]
        negatives = [_neg_sample() for _ in range(10)]

        cal = Calibrator(
            queue=mock_queue,
            config_path=validation_config,
            n_samples=10,
        )
        with patch.object(cal, "_collect_positive_samples", return_value=positives), \
             patch.object(cal, "_build_negative_samples", return_value=negatives):
            result = cal.run()

        assert 0.0 <= result.structural_threshold <= 1.0
        assert 0.0 <= result.nlp_threshold <= 1.0
        assert 0.0 <= result.arcface_threshold <= 1.0
        assert 0.0 <= result.structural_precision <= 1.0
        assert 0.0 <= result.structural_recall <= 1.0

    def test_empty_queue_returns_zero_positives(self, mock_queue, validation_config):
        mock_queue.dequeue_batch.return_value = []
        cal = Calibrator(
            queue=mock_queue,
            config_path=validation_config,
            n_samples=50,
        )
        # Should complete without error even with empty queue
        with patch.object(cal, "_build_negative_samples", return_value=[]):
            result = cal.run()
        assert result.n_positives == 0
