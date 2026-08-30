"""Pipeline orchestrator — processes all validated pairs end-to-end with resume support."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

import cv2
import yaml
from loguru import logger
from tqdm import tqdm

from pipeline.quality.scorer import QualityScorer, QualityResult
from pipeline.detection.detector import FaceDetector, DetectionResult
from pipeline.landmarks.extractor import LandmarkExtractor, LandmarkResult
from pipeline.alignment.normaliser import FaceAligner, AlignmentResult
from pipeline.alignment.pose import PoseValidator, PairPoseResult
from pipeline.segmentation.zone_mapper import ZoneMapper, ZoneMappingResult


@dataclass
class PairRecord:
    """Minimal representation of a before/after pair to be processed."""
    pair_id: int
    before_path: str
    after_path: str
    source_name: str = ""
    treatment_category: str = ""


@dataclass
class PipelineStats:
    total: int = 0
    quality_fail: int = 0
    detection_fail: int = 0
    landmark_fail: int = 0
    alignment_fail: int = 0
    pose_fail: int = 0
    passed: int = 0
    skipped_already_processed: int = 0
    elapsed_seconds: float = 0.0

    @property
    def pass_rate(self) -> float:
        processed = self.total - self.skipped_already_processed
        return self.passed / processed if processed > 0 else 0.0


@dataclass
class ProcessedPair:
    pair_id: int
    before_aligned_path: str
    after_aligned_path: str
    before_quality: dict
    after_quality: dict
    before_landmarks: list[dict]  # list of {index, x, y, z}
    after_landmarks: list[dict]
    pose_result: dict
    zones: list[dict]  # list of {zone_code, confidence}
    passed: bool
    reject_reason: str | None


class PipelineRunner:
    """
    End-to-end pipeline that processes raw downloaded image pairs into:
      - Quality-scored, aligned 512×512 images
      - 478 MediaPipe landmarks per image
      - Head pose validation
      - Zone labels

    Supports resume: writes a progress file and skips already-processed pair IDs.

    Usage:
        runner = PipelineRunner(
            aligned_output_dir="data/aligned",
            progress_file="data/pipeline_progress.json",
        )
        stats = runner.run(pairs)
    """

    def __init__(
        self,
        aligned_output_dir: str | Path = "data/aligned",
        progress_file: str | Path = "data/pipeline_progress.json",
        config_path: str | Path = "configs/pipeline.yaml",
    ) -> None:
        self._output_dir = Path(aligned_output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._progress_file = Path(progress_file)
        self._config_path = Path(config_path)

        self._scorer = QualityScorer(config_path)
        self._detector = FaceDetector(config_path)
        self._extractor = LandmarkExtractor(config_path)
        self._aligner = FaceAligner(config_path)
        self._pose_validator = PoseValidator(config_path)
        self._zone_mapper = ZoneMapper()  # always uses configs/zones.yaml

        self._processed_ids: set[int] = self._load_progress()

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, pairs: list[PairRecord]) -> PipelineStats:
        """
        Process all pairs. Already-processed pair IDs are skipped.
        Returns PipelineStats summary.
        """
        stats = PipelineStats(total=len(pairs))
        start = time.monotonic()

        for pair in tqdm(pairs, desc="Pipeline", unit="pair"):
            if pair.pair_id in self._processed_ids:
                stats.skipped_already_processed += 1
                continue

            result = self._process_pair(pair)

            if result.passed:
                stats.passed += 1
            elif result.reject_reason:
                reason = result.reject_reason.split(":")[0]
                if "quality" in reason:
                    stats.quality_fail += 1
                elif "detection" in reason or "face" in reason:
                    stats.detection_fail += 1
                elif "landmark" in reason:
                    stats.landmark_fail += 1
                elif "alignment" in reason:
                    stats.alignment_fail += 1
                elif "pose" in reason:
                    stats.pose_fail += 1

            self._processed_ids.add(pair.pair_id)
            self._save_progress()

        stats.elapsed_seconds = time.monotonic() - start
        logger.info(
            f"[runner] Done: {stats.passed}/{stats.total} passed "
            f"({stats.pass_rate:.1%}) in {stats.elapsed_seconds:.1f}s"
        )
        return stats

    # ── Single pair processing ─────────────────────────────────────────────────

    def _process_pair(self, pair: PairRecord) -> ProcessedPair:
        """Run the full pipeline for one before/after pair."""
        before_img = cv2.imread(pair.before_path)
        after_img = cv2.imread(pair.after_path)

        if before_img is None or after_img is None:
            return ProcessedPair(
                pair_id=pair.pair_id,
                before_aligned_path="", after_aligned_path="",
                before_quality={}, after_quality={},
                before_landmarks=[], after_landmarks=[],
                pose_result={}, zones=[],
                passed=False, reject_reason="quality:unreadable_image",
            )

        # Step 1 — Quality scoring
        bq = self._scorer.score_array(before_img)
        aq = self._scorer.score_array(after_img)
        if bq.grade == "FAIL":
            return self._reject(pair, f"quality:before:{bq.fail_reason}", bq, aq)
        if aq.grade == "FAIL":
            return self._reject(pair, f"quality:after:{aq.fail_reason}", bq, aq)

        # Step 2 — Face detection
        b_det = self._detector.detect(before_img)
        a_det = self._detector.detect(after_img)
        if not b_det.accepted:
            return self._reject(pair, f"detection:before:{b_det.reject_reason}", bq, aq)
        if not a_det.accepted:
            return self._reject(pair, f"detection:after:{a_det.reject_reason}", bq, aq)

        # Step 3 — Landmark extraction
        b_lm = self._extractor.extract(before_img)
        a_lm = self._extractor.extract(after_img)
        if not b_lm.success:
            return self._reject(pair, "landmark:before:no_face_mesh", bq, aq)
        if not a_lm.success:
            return self._reject(pair, "landmark:after:no_face_mesh", bq, aq)

        # Step 4 — Alignment
        h, w = before_img.shape[:2]
        b_align = self._aligner.align(before_img, b_lm)
        a_align = self._aligner.align(after_img, a_lm)
        if not b_align.success:
            return self._reject(pair, f"alignment:before:{b_align.reject_reason}", bq, aq)
        if not a_align.success:
            return self._reject(pair, f"alignment:after:{a_align.reject_reason}", bq, aq)

        # Step 5 — Pose validation
        pose_result = self._pose_validator.validate_pair(b_lm, a_lm, w, h)
        if not pose_result.accepted:
            return self._reject(pair, f"pose:{pose_result.reject_reason}", bq, aq)

        # Step 6 — Zone mapping
        zone_result = self._zone_mapper.map(b_lm)

        # Write aligned images
        b_out = self._aligned_path(pair.pair_id, "before")
        a_out = self._aligned_path(pair.pair_id, "after")
        cv2.imwrite(str(b_out), b_align.aligned_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(a_out), a_align.aligned_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return ProcessedPair(
            pair_id=pair.pair_id,
            before_aligned_path=str(b_out),
            after_aligned_path=str(a_out),
            before_quality=_quality_to_dict(bq),
            after_quality=_quality_to_dict(aq),
            before_landmarks=_landmarks_to_list(b_lm),
            after_landmarks=_landmarks_to_list(a_lm),
            pose_result=_pose_to_dict(pose_result),
            zones=[{"zone_code": z.zone_code, "confidence": z.confidence} for z in zone_result.zones],
            passed=True,
            reject_reason=None,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _reject(
        self, pair: PairRecord, reason: str,
        bq: QualityResult | None = None, aq: QualityResult | None = None
    ) -> ProcessedPair:
        logger.debug(f"[runner] Pair {pair.pair_id} rejected: {reason}")
        return ProcessedPair(
            pair_id=pair.pair_id,
            before_aligned_path="", after_aligned_path="",
            before_quality=_quality_to_dict(bq) if bq else {},
            after_quality=_quality_to_dict(aq) if aq else {},
            before_landmarks=[], after_landmarks=[],
            pose_result={}, zones=[],
            passed=False, reject_reason=reason,
        )

    def _aligned_path(self, pair_id: int, role: str) -> Path:
        shard = f"{pair_id % 100:02d}"
        p = self._output_dir / shard
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{pair_id}_{role}.jpg"

    def _load_progress(self) -> set[int]:
        if self._progress_file.exists():
            try:
                data = json.loads(self._progress_file.read_text())
                return set(data.get("processed_ids", []))
            except Exception:
                return set()
        return set()

    def _save_progress(self) -> None:
        self._progress_file.parent.mkdir(parents=True, exist_ok=True)
        self._progress_file.write_text(
            json.dumps({"processed_ids": sorted(self._processed_ids)})
        )


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _quality_to_dict(q: QualityResult | None) -> dict:
    if q is None:
        return {}
    return {
        "blur_score": q.blur_score,
        "lighting_uniformity": q.lighting_uniformity,
        "mean_brightness": q.mean_brightness,
        "width": q.width,
        "height": q.height,
        "resolution_pass": q.resolution_pass,
        "grade": q.grade,
        "fail_reason": q.fail_reason,
    }


def _landmarks_to_list(lm: LandmarkResult) -> list[dict]:
    return [{"index": p.index, "x": p.x, "y": p.y, "z": p.z} for p in lm.landmarks]


def _pose_to_dict(p: PairPoseResult) -> dict:
    result: dict = {"accepted": p.accepted, "reject_reason": p.reject_reason}
    if p.before_pose:
        result["before"] = {"yaw": p.before_pose.yaw, "pitch": p.before_pose.pitch, "roll": p.before_pose.roll}
    if p.after_pose:
        result["after"] = {"yaw": p.after_pose.yaw, "pitch": p.after_pose.pitch, "roll": p.after_pose.roll}
    return result
