"""Head pose estimator — yaw/pitch/roll from MediaPipe landmarks via solvePnP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger

from pipeline.landmarks.extractor import LandmarkResult


# 3D reference points for 6 key facial landmarks in a canonical face model
# (nose tip, chin, left eye outer, right eye outer, left mouth, right mouth)
# Units: mm in a canonical face coordinate system
_FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0],    # nose tip (origin)
    [0.0,   -63.6, -12.5],   # chin
    [-43.3,  32.7, -26.0],   # left eye outer corner
    [43.3,   32.7, -26.0],   # right eye outer corner
    [-28.9, -28.9, -24.1],   # left mouth corner
    [28.9,  -28.9, -24.1],   # right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
_LANDMARK_IDS = [
    1,    # nose tip
    152,  # chin
    33,   # left eye outer corner
    263,  # right eye outer corner
    61,   # left mouth corner
    291,  # right mouth corner
]


@dataclass
class PoseAngles:
    yaw: float    # degrees — positive = face turned right
    pitch: float  # degrees — positive = face tilted up
    roll: float   # degrees — positive = face tilted left


@dataclass
class PoseResult:
    pose: PoseAngles | None
    success: bool
    reject_reason: str | None = None


@dataclass
class PairPoseResult:
    before_pose: PoseAngles | None
    after_pose: PoseAngles | None
    accepted: bool
    reject_reason: str | None = None


class PoseValidator:
    """
    Estimates head pose (yaw/pitch/roll) from MediaPipe landmarks using solvePnP.
    Validates that a before/after pair has consistent pose within configured limits.
    Config read from configs/pipeline.yaml.
    """

    def __init__(self, config_path: str | Path = "configs/pipeline.yaml") -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        pv = cfg["pair_validation"]
        self._max_yaw: float = pv["max_yaw_difference_deg"]
        self._max_pitch: float = pv["max_pitch_difference_deg"]
        self._max_roll: float = pv["max_roll_difference_deg"]

    def estimate_pose(
        self, landmark_result: LandmarkResult, image_width: int, image_height: int
    ) -> PoseResult:
        """
        Estimate head pose from landmarks.
        Returns PoseResult with success=False if estimation fails.
        """
        if not landmark_result.success or len(landmark_result.landmarks) < max(_LANDMARK_IDS) + 1:
            return PoseResult(pose=None, success=False, reject_reason="insufficient_landmarks")

        lm_map = {lm.index: lm for lm in landmark_result.landmarks}
        missing = [i for i in _LANDMARK_IDS if i not in lm_map]
        if missing:
            return PoseResult(pose=None, success=False, reject_reason=f"missing_landmarks:{missing}")

        image_2d = np.array([
            [lm_map[i].x * image_width, lm_map[i].y * image_height]
            for i in _LANDMARK_IDS
        ], dtype=np.float64)

        focal = float(image_width)
        cx, cy = image_width / 2.0, image_height / 2.0
        camera_matrix = np.array([
            [focal, 0, cx],
            [0, focal, cy],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            _FACE_3D_MODEL, image_2d, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return PoseResult(pose=None, success=False, reject_reason="solvepnp_failed")

        rmat, _ = cv2.Rodrigues(rvec)
        angles = _rotation_matrix_to_euler(rmat)
        return PoseResult(
            pose=PoseAngles(yaw=angles[1], pitch=angles[0], roll=angles[2]),
            success=True,
        )

    def validate_pair(
        self,
        before_landmarks: LandmarkResult,
        after_landmarks: LandmarkResult,
        image_width: int,
        image_height: int,
    ) -> PairPoseResult:
        """
        Validate that a before/after pair has sufficiently similar head pose.
        Returns PairPoseResult with accepted=True only if both poses are within limits.
        """
        before_pose_result = self.estimate_pose(before_landmarks, image_width, image_height)
        after_pose_result = self.estimate_pose(after_landmarks, image_width, image_height)

        if not before_pose_result.success:
            return PairPoseResult(
                before_pose=None, after_pose=None, accepted=False,
                reject_reason=f"before_pose_failed:{before_pose_result.reject_reason}",
            )
        if not after_pose_result.success:
            return PairPoseResult(
                before_pose=before_pose_result.pose, after_pose=None, accepted=False,
                reject_reason=f"after_pose_failed:{after_pose_result.reject_reason}",
            )

        bp = before_pose_result.pose
        ap = after_pose_result.pose

        yaw_diff = abs(bp.yaw - ap.yaw)
        pitch_diff = abs(bp.pitch - ap.pitch)
        roll_diff = abs(bp.roll - ap.roll)

        if yaw_diff > self._max_yaw:
            return PairPoseResult(
                before_pose=bp, after_pose=ap, accepted=False,
                reject_reason=f"yaw_mismatch:{yaw_diff:.1f}>{self._max_yaw}",
            )
        if pitch_diff > self._max_pitch:
            return PairPoseResult(
                before_pose=bp, after_pose=ap, accepted=False,
                reject_reason=f"pitch_mismatch:{pitch_diff:.1f}>{self._max_pitch}",
            )
        if roll_diff > self._max_roll:
            return PairPoseResult(
                before_pose=bp, after_pose=ap, accepted=False,
                reject_reason=f"roll_mismatch:{roll_diff:.1f}>{self._max_roll}",
            )

        return PairPoseResult(before_pose=bp, after_pose=ap, accepted=True)


def _rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Convert 3×3 rotation matrix to Euler angles (pitch, yaw, roll) in degrees."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    return float(np.degrees(pitch)), float(np.degrees(yaw)), float(np.degrees(roll))
