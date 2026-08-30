"""Tests for face alignment normaliser."""

import numpy as np
import pytest
import cv2
from unittest.mock import patch, MagicMock
from pipeline.alignment.normaliser import FaceAligner, AlignmentResult
from pipeline.landmarks.extractor import LandmarkResult, LandmarkPoint


CONFIG_PATH = "configs/pipeline.yaml"


def make_landmark_result_with_iris(
    left_x=0.33, left_y=0.40,
    right_x=0.67, right_y=0.40,
    n_base=478,
):
    """Build a LandmarkResult with iris landmarks at given normalised positions."""
    lms = [LandmarkPoint(index=i, x=float(i) / 500, y=0.5, z=0.0) for i in range(n_base)]
    lms.append(LandmarkPoint(index=468, x=left_x, y=left_y, z=0.0))
    lms.append(LandmarkPoint(index=473, x=right_x, y=right_y, z=0.0))
    return LandmarkResult(landmarks=lms, success=True, num_landmarks=len(lms))


@pytest.fixture
def aligner():
    return FaceAligner(CONFIG_PATH)


@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)


@pytest.fixture
def good_landmarks():
    return make_landmark_result_with_iris(left_x=0.33, left_y=0.40, right_x=0.67, right_y=0.40)


def test_align_produces_correct_size(aligner, sample_image, good_landmarks):
    result = aligner.align(sample_image, good_landmarks)
    assert result.success is True
    assert result.aligned_image is not None
    assert result.aligned_image.shape == (512, 512, 3)


def test_align_target_size_property(aligner):
    assert aligner.target_size == 512


def test_align_returns_float_angle(aligner, sample_image, good_landmarks):
    result = aligner.align(sample_image, good_landmarks)
    assert isinstance(result.angle_deg, float)


def test_align_returns_positive_scale(aligner, sample_image, good_landmarks):
    result = aligner.align(sample_image, good_landmarks)
    assert result.scale > 0


def test_align_fails_without_landmarks(aligner, sample_image):
    empty_lm = LandmarkResult(landmarks=[], success=False, num_landmarks=0)
    result = aligner.align(sample_image, empty_lm)
    assert result.success is False
    assert result.reject_reason == "no_eye_landmarks"


def test_align_fails_when_eyes_too_close(aligner, sample_image):
    # Eyes at nearly the same position → distance < 5 pixels
    lm = make_landmark_result_with_iris(left_x=0.50, left_y=0.40, right_x=0.501, right_y=0.40)
    result = aligner.align(sample_image, lm)
    assert result.success is False
    assert result.reject_reason == "eye_distance_too_small"


def test_align_file_writes_output(aligner, good_landmarks, tmp_path):
    img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    src = tmp_path / "input.jpg"
    cv2.imwrite(str(src), img)
    out = tmp_path / "aligned.jpg"
    result = aligner.align_file(src, good_landmarks, output_path=out)
    assert result.success is True
    assert out.exists()


def test_align_file_unreadable_returns_fail(aligner, good_landmarks, tmp_path):
    result = aligner.align_file(tmp_path / "nonexistent.jpg", good_landmarks)
    assert result.success is False
    assert result.reject_reason == "unreadable"


def test_aligned_image_is_bgr(aligner, sample_image, good_landmarks):
    result = aligner.align(sample_image, good_landmarks)
    assert result.aligned_image.ndim == 3
    assert result.aligned_image.shape[2] == 3
