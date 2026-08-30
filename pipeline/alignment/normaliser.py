"""Face alignment — affine warp to canonical eye-distance position at 512×512."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger

from pipeline.landmarks.extractor import LandmarkResult


@dataclass
class AlignmentResult:
    aligned_image: np.ndarray | None  # BGR, target_size × target_size
    success: bool
    scale: float = 1.0
    angle_deg: float = 0.0
    reject_reason: str | None = None


class FaceAligner:
    """
    Aligns a face image so that:
      - Eyes are on a horizontal axis (rotation corrected)
      - Eye distance = eye_distance_ratio × target_size
      - Face is centred with margin padding
      - Output is target_size × target_size BGR image

    Config read from configs/pipeline.yaml.
    """

    def __init__(self, config_path: str | Path = "configs/pipeline.yaml") -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        align = cfg["alignment"]
        self._target: int = align["target_size"]
        self._eye_ratio: float = align["eye_distance_ratio"]
        self._margin: float = align["margin"]

    def align(self, image: np.ndarray, landmark_result: LandmarkResult) -> AlignmentResult:
        """
        Align a BGR image using pre-computed landmarks.
        Returns AlignmentResult with success=False if alignment is not possible.
        """
        from pipeline.landmarks.extractor import LandmarkExtractor
        eye_centers = LandmarkExtractor.eye_centers(landmark_result)

        if eye_centers is None:
            return AlignmentResult(aligned_image=None, success=False, reject_reason="no_eye_landmarks")

        left_eye, right_eye = eye_centers
        h, w = image.shape[:2]

        # Convert normalised coords to pixels
        lx, ly = left_eye[0] * w, left_eye[1] * h
        rx, ry = right_eye[0] * w, right_eye[1] * h

        # Compute rotation angle to level eyes
        dy = ry - ly
        dx = rx - lx
        angle = float(np.degrees(np.arctan2(dy, dx)))

        # Current and target eye distance
        eye_dist = float(np.sqrt(dx ** 2 + dy ** 2))
        if eye_dist < 5:
            return AlignmentResult(aligned_image=None, success=False, reject_reason="eye_distance_too_small")

        target_eye_dist = self._eye_ratio * self._target
        scale = target_eye_dist / eye_dist

        # Eye midpoint → will become the image centre after affine transform
        eye_mid_x = (lx + rx) / 2.0
        eye_mid_y = (ly + ry) / 2.0

        # Build rotation + scale matrix around the eye midpoint
        M = cv2.getRotationMatrix2D((eye_mid_x, eye_mid_y), angle, scale)

        # Add translation so eye midpoint maps to (target/2, target * 0.35)
        # 0.35 from top leaves room for forehead (1-0.35=0.65 below eyes)
        target_cx = self._target / 2.0
        target_cy = self._target * 0.35
        M[0, 2] += target_cx - eye_mid_x
        M[1, 2] += target_cy - eye_mid_y

        # Warp
        aligned = cv2.warpAffine(
            image, M, (self._target, self._target),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return AlignmentResult(
            aligned_image=aligned,
            success=True,
            scale=scale,
            angle_deg=angle,
        )

    def align_file(
        self,
        image_path: str | Path,
        landmark_result: LandmarkResult,
        output_path: str | Path | None = None,
    ) -> AlignmentResult:
        """
        Align an image file and optionally write the result.
        output_path: if provided, writes the aligned JPEG to this path.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return AlignmentResult(aligned_image=None, success=False, reject_reason="unreadable")

        result = self.align(img, landmark_result)

        if result.success and output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), result.aligned_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            logger.debug(f"[aligner] Wrote aligned image to {output_path}")

        return result

    @property
    def target_size(self) -> int:
        return self._target
