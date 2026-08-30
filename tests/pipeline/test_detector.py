"""Tests for the face detector (mocked — no real model weights needed)."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pipeline.detection.detector import FaceDetector, FaceBox, DetectionResult


CONFIG_PATH = "configs/pipeline.yaml"


def make_face_box(x1=50, y1=50, x2=200, y2=200, conf=0.97):
    return FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)


@pytest.fixture
def detector():
    return FaceDetector(CONFIG_PATH)


def mock_detect(result_faces, method="mtcnn"):
    """Return a DetectionResult with given faces (before acceptance policy)."""
    return DetectionResult(faces=result_faces, method=method, accepted=False)


def test_acceptance_single_good_face(detector):
    faces = [make_face_box(conf=0.97)]
    result = detector._apply_acceptance_policy(DetectionResult(faces=faces, method="mtcnn", accepted=False))
    assert result.accepted is True
    assert result.reject_reason is None


def test_rejection_too_many_faces(detector):
    faces = [make_face_box(conf=0.97), make_face_box(x1=250, x2=400, conf=0.95)]
    result = detector._apply_acceptance_policy(DetectionResult(faces=faces, method="mtcnn", accepted=False))
    assert result.accepted is False
    assert "too_many_faces" in result.reject_reason


def test_rejection_no_faces(detector):
    result = detector._apply_acceptance_policy(DetectionResult(faces=[], method="mtcnn", accepted=False))
    assert result.accepted is False
    assert result.reject_reason == "no_face_detected"


def test_rejection_face_too_small(detector):
    # Face box smaller than min_face_size_px (80px from config)
    faces = [make_face_box(x1=0, y1=0, x2=30, y2=30, conf=0.97)]
    result = detector._apply_acceptance_policy(DetectionResult(faces=faces, method="mtcnn", accepted=False))
    assert result.accepted is False
    assert result.reject_reason == "no_face_detected"


def test_rejection_low_confidence_face(detector):
    # Low confidence face — filtered out during MTCNN/RetinaFace parsing, not here
    # After filtering, if nothing remains → no_face_detected
    faces = []  # low-conf faces already filtered before acceptance policy
    result = detector._apply_acceptance_policy(DetectionResult(faces=faces, method="mtcnn", accepted=False))
    assert result.accepted is False


def test_face_box_dimensions():
    box = make_face_box(x1=10, y1=20, x2=110, y2=120)
    assert box.width == 100
    assert box.height == 100
    assert box.area == 10000


def test_largest_face_selected(detector):
    small = make_face_box(x1=0, y1=0, x2=50, y2=50)
    large = make_face_box(x1=50, y1=50, x2=250, y2=250)
    result = DetectionResult(faces=[small, large], method="mtcnn", accepted=True)
    assert detector.largest_face(result) is large


def test_largest_face_empty_returns_none(detector):
    result = DetectionResult(faces=[], method="none", accepted=False)
    assert detector.largest_face(result) is None


def test_unreadable_file_returns_fail(detector, tmp_path):
    result = detector.detect_file(tmp_path / "nonexistent.jpg")
    assert result.accepted is False
    assert result.reject_reason == "unreadable"


@patch("pipeline.detection.detector._load_mtcnn")
def test_mtcnn_exception_returns_empty(mock_load, detector):
    mock_mtcnn = MagicMock()
    mock_mtcnn.detect.side_effect = RuntimeError("GPU error")
    mock_load.return_value = mock_mtcnn
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    result = detector._detect_mtcnn(img)
    assert result.faces == []
    assert result.method == "mtcnn"
