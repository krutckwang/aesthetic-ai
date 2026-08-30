"""Tests for pipeline quality scorer."""

import numpy as np
import pytest
import cv2
from unittest.mock import patch, MagicMock
from pipeline.quality.scorer import QualityScorer, QualityResult


CONFIG_PATH = "configs/pipeline.yaml"


@pytest.fixture
def scorer():
    return QualityScorer(CONFIG_PATH)


def make_image(h=300, w=300, brightness=128, blur_sigma=0):
    """Create a synthetic BGR image with optional gaussian blur."""
    img = np.full((h, w, 3), brightness, dtype=np.uint8)
    # Add checkerboard pattern for texture (increases Laplacian variance)
    for i in range(0, h, 20):
        for j in range(0, w, 20):
            if (i // 20 + j // 20) % 2 == 0:
                img[i:i+20, j:j+20] = min(255, brightness + 60)
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (31, 31), blur_sigma)
    return img


def test_sharp_image_passes(scorer):
    img = make_image(h=400, w=400, brightness=128)
    result = scorer.score_array(img)
    assert result.grade == "PASS"
    assert result.blur_score > 0


def test_blurry_image_fails(scorer):
    img = make_image(h=400, w=400, brightness=128, blur_sigma=20)
    result = scorer.score_array(img)
    assert result.grade == "FAIL"
    assert result.fail_reason is not None
    assert "blur" in result.fail_reason


def test_small_image_fails_resolution(scorer):
    img = make_image(h=100, w=100)
    result = scorer.score_array(img)
    assert result.grade == "FAIL"
    assert "resolution" in result.fail_reason


def test_overexposed_image_fails(scorer):
    img = make_image(h=400, w=400, brightness=250)
    result = scorer.score_array(img)
    # May fail on overexposure or blur depending on how uniform the bright image is
    assert result.grade == "FAIL"


def test_underexposed_image_fails(scorer):
    img = make_image(h=400, w=400, brightness=5, blur_sigma=0)
    result = scorer.score_array(img)
    assert result.grade == "FAIL"


def test_result_has_dimensions(scorer):
    img = make_image(h=400, w=300)
    result = scorer.score_array(img)
    assert result.width == 300
    assert result.height == 400


def test_empty_array_fails(scorer):
    result = scorer.score_array(None)
    assert result.grade == "FAIL"
    assert result.fail_reason == "empty_array"


def test_unreadable_file_fails(scorer, tmp_path):
    result = scorer.score(tmp_path / "nonexistent.jpg")
    assert result.grade == "FAIL"
    assert result.fail_reason == "unreadable"


def test_blur_score_decreases_with_blur(scorer):
    sharp = scorer.score_array(make_image(400, 400, 128, blur_sigma=0))
    blurry = scorer.score_array(make_image(400, 400, 128, blur_sigma=15))
    assert sharp.blur_score > blurry.blur_score


def test_lighting_uniformity_range(scorer):
    img = make_image(h=400, w=400, brightness=128)
    result = scorer.score_array(img)
    assert 0.0 <= result.lighting_uniformity <= 1.0


def test_score_file_reads_from_disk(scorer, tmp_path):
    img = make_image(h=400, w=400, brightness=128)
    path = tmp_path / "test.jpg"
    cv2.imwrite(str(path), img)
    result = scorer.score(path)
    assert result.width == 400
    assert result.height == 400
