"""MediaPipe 478-point face mesh landmark extractor."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from loguru import logger


@dataclass
class LandmarkPoint:
    index: int
    x: float  # normalised 0–1
    y: float  # normalised 0–1
    z: float  # relative depth


@dataclass
class LandmarkResult:
    landmarks: list[LandmarkPoint]  # 478 points when successful
    success: bool
    num_landmarks: int
    method: str = "mediapipe"

    def as_xy_array(self) -> np.ndarray:
        """Return (N, 2) array of (x, y) normalised coordinates."""
        return np.array([[lm.x, lm.y] for lm in self.landmarks], dtype=np.float32)

    def as_xyz_array(self) -> np.ndarray:
        """Return (N, 3) array of (x, y, z) coordinates."""
        return np.array([[lm.x, lm.y, lm.z] for lm in self.landmarks], dtype=np.float32)

    def pixel_xy(self, image_width: int, image_height: int) -> np.ndarray:
        """Return (N, 2) array of pixel coordinates."""
        xy = self.as_xy_array()
        xy[:, 0] *= image_width
        xy[:, 1] *= image_height
        return xy


class LandmarkExtractor:
    """
    Extracts 478 MediaPipe face mesh landmarks from an image.
    Config read from configs/pipeline.yaml.
    """

    def __init__(self, config_path: str | Path = "configs/pipeline.yaml") -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        lm = cfg["landmark_extraction"]
        self._min_detection: float = lm["min_detection_confidence"]
        self._min_tracking: float = lm["min_tracking_confidence"]
        self._refine: bool = lm["refine_landmarks"]
        self._num_landmarks: int = lm["num_landmarks"]

    def extract(self, image: np.ndarray) -> LandmarkResult:
        """
        Extract landmarks from a BGR numpy image.
        Returns LandmarkResult with success=False if no face found.
        """
        try:
            face_mesh = _load_face_mesh(
                self._min_detection, self._min_tracking, self._refine
            )
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                return LandmarkResult(landmarks=[], success=False, num_landmarks=0)

            raw = results.multi_face_landmarks[0].landmark
            landmarks = [
                LandmarkPoint(index=i, x=lm.x, y=lm.y, z=lm.z)
                for i, lm in enumerate(raw)
            ]
            return LandmarkResult(
                landmarks=landmarks,
                success=True,
                num_landmarks=len(landmarks),
            )

        except Exception as exc:
            logger.warning(f"[landmarks] MediaPipe error: {exc}")
            return LandmarkResult(landmarks=[], success=False, num_landmarks=0)

    def extract_file(self, image_path: str | Path) -> LandmarkResult:
        """Extract landmarks from an image file."""
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"[landmarks] Cannot read: {image_path}")
            return LandmarkResult(landmarks=[], success=False, num_landmarks=0)
        return self.extract(img)

    # ── Convenience helpers ───────────────────────────────────────────────────

    @staticmethod
    def eye_centers(result: LandmarkResult) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """
        Return (left_eye_center, right_eye_center) as normalised (x, y) tuples.
        Uses iris landmarks 468 (left) and 473 (right) when refine_landmarks=True,
        falls back to eye contour mean otherwise.
        """
        if not result.success or not result.landmarks:
            return None

        lm_map = {lm.index: lm for lm in result.landmarks}

        # Iris center landmarks (available only with refine_landmarks=True)
        if 468 in lm_map and 473 in lm_map:
            left = (lm_map[468].x, lm_map[468].y)
            right = (lm_map[473].x, lm_map[473].y)
            return left, right

        # Fallback: mean of left/right eye contour landmarks
        left_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

        def mean_xy(indices: list[int]) -> tuple[float, float]:
            pts = [lm_map[i] for i in indices if i in lm_map]
            if not pts:
                return 0.5, 0.5
            return float(np.mean([p.x for p in pts])), float(np.mean([p.y for p in pts]))

        return mean_xy(left_indices), mean_xy(right_indices)


# ── Cached model loader ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_face_mesh(min_detection: float, min_tracking: float, refine: bool):
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=refine,
        min_detection_confidence=min_detection,
        min_tracking_confidence=min_tracking,
    )
