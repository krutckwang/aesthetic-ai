"""Tests for the landmark extractor (mocked MediaPipe — no model weights)."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pipeline.landmarks.extractor import LandmarkExtractor, LandmarkResult, LandmarkPoint


CONFIG_PATH = "configs/pipeline.yaml"


def make_landmark_result(n=478, success=True):
    """Build a synthetic LandmarkResult with evenly spaced landmarks."""
    if not success:
        return LandmarkResult(landmarks=[], success=False, num_landmarks=0)
    lms = [LandmarkPoint(index=i, x=float(i) / n, y=float(i) / n, z=0.0) for i in range(n)]
    return LandmarkResult(landmarks=lms, success=True, num_landmarks=n)


def make_result_with_iris(left_x=0.35, left_y=0.40, right_x=0.65, right_y=0.40):
    """Build a LandmarkResult with iris landmarks 468 and 473 set."""
    lms = [LandmarkPoint(index=i, x=float(i) / 500, y=float(i) / 500, z=0.0) for i in range(478)]
    lms.append(LandmarkPoint(index=468, x=left_x, y=left_y, z=0.0))
    lms.append(LandmarkPoint(index=473, x=right_x, y=right_y, z=0.0))
    return LandmarkResult(landmarks=lms, success=True, num_landmarks=len(lms))


@pytest.fixture
def extractor():
    return LandmarkExtractor(CONFIG_PATH)


def test_as_xy_array_shape(extractor):
    result = make_landmark_result(478)
    arr = result.as_xy_array()
    assert arr.shape == (478, 2)


def test_as_xyz_array_shape(extractor):
    result = make_landmark_result(478)
    arr = result.as_xyz_array()
    assert arr.shape == (478, 3)


def test_pixel_xy_scales_correctly(extractor):
    lms = [LandmarkPoint(index=0, x=0.5, y=0.5, z=0.0)]
    result = LandmarkResult(landmarks=lms, success=True, num_landmarks=1)
    px = result.pixel_xy(image_width=200, image_height=100)
    assert px[0, 0] == pytest.approx(100.0)
    assert px[0, 1] == pytest.approx(50.0)


def test_eye_centers_uses_iris_landmarks(extractor):
    result = make_result_with_iris(left_x=0.35, left_y=0.40, right_x=0.65, right_y=0.40)
    centers = extractor.eye_centers(result)
    assert centers is not None
    left, right = centers
    assert left == pytest.approx((0.35, 0.40))
    assert right == pytest.approx((0.65, 0.40))


def test_eye_centers_returns_none_on_failure(extractor):
    result = LandmarkResult(landmarks=[], success=False, num_landmarks=0)
    assert extractor.eye_centers(result) is None


def test_eye_centers_fallback_without_iris(extractor):
    # Build result with only standard 478 landmarks, no iris (468/473)
    lms = [LandmarkPoint(index=i, x=float(i) / 500, y=0.4, z=0.0) for i in range(478)]
    result = LandmarkResult(landmarks=lms, success=True, num_landmarks=478)
    centers = extractor.eye_centers(result)
    # Should still return something using contour fallback
    assert centers is not None
    assert len(centers) == 2


def test_unreadable_file_returns_failure(extractor, tmp_path):
    result = extractor.extract_file(tmp_path / "nonexistent.jpg")
    assert result.success is False
    assert result.num_landmarks == 0


@patch("pipeline.landmarks.extractor._load_face_mesh")
def test_mediapipe_exception_returns_failure(mock_load, extractor):
    mock_mesh = MagicMock()
    mock_mesh.process.side_effect = RuntimeError("MediaPipe error")
    mock_load.return_value = mock_mesh
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    result = extractor.extract(img)
    assert result.success is False


@patch("pipeline.landmarks.extractor._load_face_mesh")
def test_no_face_returns_failure(mock_load, extractor):
    mock_mesh = MagicMock()
    mock_result = MagicMock()
    mock_result.multi_face_landmarks = None
    mock_mesh.process.return_value = mock_result
    mock_load.return_value = mock_mesh
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    result = extractor.extract(img)
    assert result.success is False
    assert result.num_landmarks == 0
