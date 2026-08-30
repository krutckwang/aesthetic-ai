"""Tests for the pipeline orchestrator runner."""

import json
import numpy as np
import pytest
import cv2
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline.runner import PipelineRunner, PairRecord, PipelineStats, ProcessedPair
from pipeline.quality.scorer import QualityResult
from pipeline.detection.detector import DetectionResult, FaceBox
from pipeline.landmarks.extractor import LandmarkResult, LandmarkPoint
from pipeline.alignment.normaliser import AlignmentResult
from pipeline.alignment.pose import PairPoseResult, PoseAngles
from pipeline.segmentation.zone_mapper import ZoneMappingResult, ZonePresence


CONFIG_PATH = "configs/pipeline.yaml"


def make_good_quality():
    return QualityResult(
        blur_score=200.0, lighting_uniformity=0.7, mean_brightness=128.0,
        width=400, height=400, resolution_pass=True, grade="PASS",
    )


def make_bad_quality(reason="blur_too_low:50.0<80.0"):
    return QualityResult(
        blur_score=50.0, lighting_uniformity=0.7, mean_brightness=128.0,
        width=400, height=400, resolution_pass=True, grade="FAIL", fail_reason=reason,
    )


def make_good_detection():
    box = FaceBox(x1=50, y1=50, x2=200, y2=200, confidence=0.97)
    return DetectionResult(faces=[box], method="mtcnn", accepted=True)


def make_bad_detection():
    return DetectionResult(faces=[], method="mtcnn", accepted=False, reject_reason="no_face_detected")


def make_good_landmarks(n=478):
    lms = [LandmarkPoint(index=i, x=float(i) / n, y=float(i) / n, z=0.0) for i in range(n)]
    lms.append(LandmarkPoint(index=468, x=0.35, y=0.40, z=0.0))
    lms.append(LandmarkPoint(index=473, x=0.65, y=0.40, z=0.0))
    return LandmarkResult(landmarks=lms, success=True, num_landmarks=len(lms))


def make_bad_landmarks():
    return LandmarkResult(landmarks=[], success=False, num_landmarks=0)


def make_good_alignment():
    return AlignmentResult(
        aligned_image=np.zeros((512, 512, 3), dtype=np.uint8),
        success=True, scale=1.2, angle_deg=0.5,
    )


def make_bad_alignment():
    return AlignmentResult(aligned_image=None, success=False, reject_reason="eye_distance_too_small")


def make_good_pose():
    p = PoseAngles(yaw=2.0, pitch=1.0, roll=0.5)
    return PairPoseResult(before_pose=p, after_pose=p, accepted=True)


def make_bad_pose():
    return PairPoseResult(before_pose=None, after_pose=None, accepted=False, reject_reason="yaw_mismatch:20.0>15.0")


def make_good_zones():
    return ZoneMappingResult(
        zones=[ZonePresence(zone_code="lips", confidence=0.9, landmark_count=10, centroid_x=0.5, centroid_y=0.7)],
        success=True, total_zones_checked=10,
    )


def write_dummy_images(tmp_path, pair_id=1):
    """Write two dummy JPEG images that cv2 can read."""
    img = np.random.randint(100, 200, (400, 400, 3), dtype=np.uint8)
    b_path = tmp_path / f"{pair_id}_before.jpg"
    a_path = tmp_path / f"{pair_id}_after.jpg"
    cv2.imwrite(str(b_path), img)
    cv2.imwrite(str(a_path), img)
    return str(b_path), str(a_path)


@pytest.fixture
def runner(tmp_path):
    return PipelineRunner(
        aligned_output_dir=tmp_path / "aligned",
        progress_file=tmp_path / "progress.json",
        config_path=CONFIG_PATH,
    )


def test_runner_processes_good_pair(tmp_path):
    runner = PipelineRunner(
        aligned_output_dir=tmp_path / "aligned",
        progress_file=tmp_path / "progress.json",
        config_path=CONFIG_PATH,
    )
    b_path, a_path = write_dummy_images(tmp_path, pair_id=1)

    # Directly replace internal module instances with mocks
    runner._scorer = MagicMock()
    runner._scorer.score_array.return_value = make_good_quality()
    runner._detector = MagicMock()
    runner._detector.detect.return_value = make_good_detection()
    runner._extractor = MagicMock()
    runner._extractor.extract.return_value = make_good_landmarks()
    runner._aligner = MagicMock()
    runner._aligner.align.return_value = make_good_alignment()
    runner._pose_validator = MagicMock()
    runner._pose_validator.validate_pair.return_value = make_good_pose()
    runner._zone_mapper = MagicMock()
    runner._zone_mapper.map.return_value = make_good_zones()

    pair = PairRecord(pair_id=1, before_path=b_path, after_path=a_path)
    result = runner._process_pair(pair)
    assert result.passed is True
    assert result.reject_reason is None


def test_runner_rejects_bad_quality(tmp_path):
    runner = PipelineRunner(
        aligned_output_dir=tmp_path / "aligned",
        progress_file=tmp_path / "progress.json",
        config_path=CONFIG_PATH,
    )
    b_path, a_path = write_dummy_images(tmp_path, pair_id=2)

    runner._scorer = MagicMock()
    runner._scorer.score_array.return_value = make_bad_quality()

    pair = PairRecord(pair_id=2, before_path=b_path, after_path=a_path)
    result = runner._process_pair(pair)
    assert result.passed is False
    assert "quality" in result.reject_reason


def test_runner_skips_already_processed(runner, tmp_path):
    b_path, a_path = write_dummy_images(tmp_path, pair_id=99)
    runner._processed_ids.add(99)
    pairs = [PairRecord(pair_id=99, before_path=b_path, after_path=a_path)]
    stats = runner.run(pairs)
    assert stats.skipped_already_processed == 1
    assert stats.passed == 0


def test_runner_saves_progress(runner, tmp_path):
    b_path, a_path = write_dummy_images(tmp_path, pair_id=5)
    runner._processed_ids.add(5)
    runner._save_progress()

    progress_file = tmp_path / "progress.json"
    assert progress_file.exists()
    data = json.loads(progress_file.read_text())
    assert 5 in data["processed_ids"]


def test_runner_loads_progress(tmp_path):
    progress_file = tmp_path / "progress.json"
    progress_file.write_text(json.dumps({"processed_ids": [10, 20, 30]}))
    runner = PipelineRunner(
        aligned_output_dir=tmp_path / "aligned",
        progress_file=progress_file,
        config_path=CONFIG_PATH,
    )
    assert 10 in runner._processed_ids
    assert 20 in runner._processed_ids


def test_runner_rejects_unreadable_images(runner, tmp_path):
    pair = PairRecord(pair_id=7, before_path="/nonexistent/a.jpg", after_path="/nonexistent/b.jpg")
    result = runner._process_pair(pair)
    assert result.passed is False
    assert "unreadable" in result.reject_reason


def test_stats_pass_rate(runner):
    stats = PipelineStats(total=10, passed=7, skipped_already_processed=0)
    assert stats.pass_rate == pytest.approx(0.7)
