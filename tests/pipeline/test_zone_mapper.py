"""Tests for zone mapper."""

import pytest
from pipeline.segmentation.zone_mapper import ZoneMapper, ZoneMappingResult
from pipeline.landmarks.extractor import LandmarkResult, LandmarkPoint


CONFIG_PATH = "configs/zones.yaml"


def make_full_landmark_result(n=478):
    """Build a LandmarkResult with n evenly distributed landmarks."""
    lms = [LandmarkPoint(index=i, x=float(i % 50) / 50, y=float(i // 50) / 20, z=0.0) for i in range(n)]
    return LandmarkResult(landmarks=lms, success=True, num_landmarks=n)


@pytest.fixture
def mapper():
    return ZoneMapper(CONFIG_PATH)


def test_mapper_loads_zone_names(mapper):
    assert len(mapper.zone_names) > 0
    assert "forehead_lines" in mapper.zone_names
    assert "lips" in mapper.zone_names
    assert "nasolabial_folds" in mapper.zone_names


def test_map_returns_zones_on_success(mapper):
    lm = make_full_landmark_result(478)
    result = mapper.map(lm)
    assert result.success is True
    assert len(result.zones) > 0


def test_map_fails_on_empty_landmarks(mapper):
    empty = LandmarkResult(landmarks=[], success=False, num_landmarks=0)
    result = mapper.map(empty)
    assert result.success is False
    assert result.zones == []


def test_zone_confidence_in_range(mapper):
    lm = make_full_landmark_result(478)
    result = mapper.map(lm)
    for zone in result.zones:
        assert 0.0 <= zone.confidence <= 1.0


def test_zone_centroid_in_range(mapper):
    lm = make_full_landmark_result(478)
    result = mapper.map(lm)
    for zone in result.zones:
        assert 0.0 <= zone.centroid_x <= 1.0
        assert 0.0 <= zone.centroid_y <= 1.0


def test_dominant_zones_returns_top_n(mapper):
    lm = make_full_landmark_result(478)
    result = mapper.map(lm)
    dominant = mapper.dominant_zones(result, min_confidence=0.0, top_n=3)
    assert len(dominant) <= 3


def test_dominant_zones_sorted_by_confidence(mapper):
    lm = make_full_landmark_result(478)
    result = mapper.map(lm)
    dominant = mapper.dominant_zones(result, min_confidence=0.0, top_n=10)
    confidences = [z.confidence for z in dominant]
    assert confidences == sorted(confidences, reverse=True)


def test_zones_for_treatment_botox(mapper):
    zones = mapper.zones_for_treatment("botox")
    assert "forehead_lines" in zones
    assert "glabellar_complex" in zones


def test_zones_for_treatment_lip_filler(mapper):
    zones = mapper.zones_for_treatment("lip_filler")
    assert "lips" in zones


def test_zones_for_treatment_unknown(mapper):
    zones = mapper.zones_for_treatment("unknown_treatment")
    assert zones == []


def test_total_zones_checked_positive(mapper):
    lm = make_full_landmark_result(478)
    result = mapper.map(lm)
    assert result.total_zones_checked > 0
