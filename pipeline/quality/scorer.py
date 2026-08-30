"""Image quality scorer — blur, lighting, resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger


@dataclass
class QualityResult:
    blur_score: float
    lighting_uniformity: float
    mean_brightness: float
    width: int
    height: int
    resolution_pass: bool
    grade: str  # "PASS" | "FAIL"
    fail_reason: str | None = None


class QualityScorer:
    """
    Scores an image on blur, lighting, and resolution.
    All thresholds are read from configs/pipeline.yaml.
    """

    def __init__(self, config_path: str | Path = "configs/pipeline.yaml") -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        qs = cfg["quality_scoring"]
        self._min_blur: float = qs["blur"]["min_score"]
        self._min_uniformity: float = qs["lighting"]["min_uniformity"]
        self._max_brightness: float = qs["lighting"]["max_brightness"]
        self._min_brightness: float = qs["lighting"]["min_brightness"]
        self._min_width: int = qs["resolution"]["min_width"]
        self._min_height: int = qs["resolution"]["min_height"]

    def score(self, image_path: str | Path) -> QualityResult:
        """
        Score a single image file. Returns QualityResult with grade PASS or FAIL.
        Returns FAIL immediately if the file cannot be read.
        """
        image_path = Path(image_path)
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"[quality] Could not read: {image_path}")
            return QualityResult(
                blur_score=0.0,
                lighting_uniformity=0.0,
                mean_brightness=0.0,
                width=0,
                height=0,
                resolution_pass=False,
                grade="FAIL",
                fail_reason="unreadable",
            )

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = self._blur_score(gray)
        uniformity, mean_brightness = self._lighting_score(gray)
        res_pass = (w >= self._min_width) and (h >= self._min_height)

        fail_reason = self._check_failures(blur, uniformity, mean_brightness, res_pass)
        grade = "PASS" if fail_reason is None else "FAIL"

        return QualityResult(
            blur_score=blur,
            lighting_uniformity=uniformity,
            mean_brightness=mean_brightness,
            width=w,
            height=h,
            resolution_pass=res_pass,
            grade=grade,
            fail_reason=fail_reason,
        )

    def score_array(self, img_bgr: np.ndarray) -> QualityResult:
        """Score a numpy BGR image array directly."""
        if img_bgr is None or img_bgr.size == 0:
            return QualityResult(
                blur_score=0.0,
                lighting_uniformity=0.0,
                mean_brightness=0.0,
                width=0,
                height=0,
                resolution_pass=False,
                grade="FAIL",
                fail_reason="empty_array",
            )

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        blur = self._blur_score(gray)
        uniformity, mean_brightness = self._lighting_score(gray)
        res_pass = (w >= self._min_width) and (h >= self._min_height)

        fail_reason = self._check_failures(blur, uniformity, mean_brightness, res_pass)
        grade = "PASS" if fail_reason is None else "FAIL"

        return QualityResult(
            blur_score=blur,
            lighting_uniformity=uniformity,
            mean_brightness=mean_brightness,
            width=w,
            height=h,
            resolution_pass=res_pass,
            grade=grade,
            fail_reason=fail_reason,
        )

    # ── Metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def _blur_score(gray: np.ndarray) -> float:
        """Laplacian variance — higher = sharper."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _lighting_score(gray: np.ndarray) -> tuple[float, float]:
        """
        Returns (uniformity, mean_brightness).
        Uniformity: 1 - coefficient_of_variation (clamped 0–1).
        Higher uniformity = more even lighting.
        """
        mean = float(gray.mean())
        std = float(gray.std())
        if mean < 1e-6:
            return 0.0, mean
        cv = std / mean  # coefficient of variation
        uniformity = max(0.0, 1.0 - cv)
        return uniformity, mean

    # ── Pass/fail logic ───────────────────────────────────────────────────────

    def _check_failures(
        self,
        blur: float,
        uniformity: float,
        mean_brightness: float,
        res_pass: bool,
    ) -> str | None:
        if not res_pass:
            return "resolution_too_small"
        if blur < self._min_blur:
            return f"blur_too_low:{blur:.1f}<{self._min_blur}"
        if uniformity < self._min_uniformity:
            return f"lighting_uneven:{uniformity:.2f}<{self._min_uniformity}"
        if mean_brightness > self._max_brightness:
            return f"overexposed:{mean_brightness:.1f}>{self._max_brightness}"
        if mean_brightness < self._min_brightness:
            return f"underexposed:{mean_brightness:.1f}<{self._min_brightness}"
        return None
