"""Face detector — MTCNN primary, InsightFace/RetinaFace fallback."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger


@dataclass
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    # 5 keypoints: left_eye, right_eye, nose, left_mouth, right_mouth
    keypoints: dict[str, tuple[float, float]] | None = None

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class DetectionResult:
    faces: list[FaceBox]
    method: str  # "mtcnn" | "retinaface" | "none"
    accepted: bool  # True if exactly 1 face meeting confidence threshold
    reject_reason: str | None = None


class FaceDetector:
    """
    Detects faces using MTCNN (primary) with InsightFace RetinaFace as fallback.
    Rejects images with zero faces or more than max_faces_allowed.
    All thresholds read from configs/pipeline.yaml.
    """

    def __init__(self, config_path: str | Path = "configs/pipeline.yaml") -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        fd = cfg["face_detection"]
        self._min_confidence: float = fd["min_face_confidence"]
        self._max_faces: int = fd["max_faces_allowed"]
        self._min_face_size: int = fd["min_face_size_px"]
        self._primary: str = fd["primary"]
        self._fallback: str = fd["fallback"]

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Detect faces in a BGR numpy image.
        Tries primary detector first, falls back if no face found.
        """
        # Try primary
        if self._primary == "mtcnn":
            result = self._detect_mtcnn(image)
        else:
            result = self._detect_retinaface(image)

        if not result.faces and self._fallback:
            logger.debug("[detector] Primary found no faces — trying fallback")
            if self._fallback == "retinaface":
                result = self._detect_retinaface(image)
            else:
                result = self._detect_mtcnn(image)

        return self._apply_acceptance_policy(result)

    def detect_file(self, image_path: str | Path) -> DetectionResult:
        """Detect faces in an image file."""
        img = cv2.imread(str(image_path))
        if img is None:
            return DetectionResult(faces=[], method="none", accepted=False, reject_reason="unreadable")
        return self.detect(img)

    # ── MTCNN ─────────────────────────────────────────────────────────────────

    def _detect_mtcnn(self, image: np.ndarray) -> DetectionResult:
        try:
            mtcnn = _load_mtcnn()
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(rgb)
            boxes, probs, points = mtcnn.detect(pil_img, landmarks=True)

            faces: list[FaceBox] = []
            if boxes is None:
                return DetectionResult(faces=[], method="mtcnn", accepted=False)

            for box, prob, kp in zip(boxes, probs, points if points is not None else [None] * len(boxes)):
                if prob is None or prob < self._min_confidence:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                keypoints = None
                if kp is not None:
                    keypoints = {
                        "left_eye": (float(kp[0][0]), float(kp[0][1])),
                        "right_eye": (float(kp[1][0]), float(kp[1][1])),
                        "nose": (float(kp[2][0]), float(kp[2][1])),
                        "left_mouth": (float(kp[3][0]), float(kp[3][1])),
                        "right_mouth": (float(kp[4][0]), float(kp[4][1])),
                    }
                faces.append(FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=float(prob), keypoints=keypoints))

            return DetectionResult(faces=faces, method="mtcnn", accepted=False)

        except Exception as exc:
            logger.warning(f"[detector] MTCNN error: {exc}")
            return DetectionResult(faces=[], method="mtcnn", accepted=False)

    # ── RetinaFace (InsightFace) ───────────────────────────────────────────────

    def _detect_retinaface(self, image: np.ndarray) -> DetectionResult:
        try:
            app = _load_insightface()
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ifaces = app.get(rgb)

            faces: list[FaceBox] = []
            for f in ifaces:
                if f.det_score < self._min_confidence:
                    continue
                x1, y1, x2, y2 = [int(v) for v in f.bbox]
                keypoints = None
                if f.kps is not None:
                    kp = f.kps
                    keypoints = {
                        "left_eye": (float(kp[0][0]), float(kp[0][1])),
                        "right_eye": (float(kp[1][0]), float(kp[1][1])),
                        "nose": (float(kp[2][0]), float(kp[2][1])),
                        "left_mouth": (float(kp[3][0]), float(kp[3][1])),
                        "right_mouth": (float(kp[4][0]), float(kp[4][1])),
                    }
                faces.append(FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=float(f.det_score), keypoints=keypoints))

            return DetectionResult(faces=faces, method="retinaface", accepted=False)

        except Exception as exc:
            logger.warning(f"[detector] RetinaFace error: {exc}")
            return DetectionResult(faces=[], method="retinaface", accepted=False)

    # ── Acceptance policy ─────────────────────────────────────────────────────

    def _apply_acceptance_policy(self, result: DetectionResult) -> DetectionResult:
        """Filter by size and apply max_faces policy. Sets accepted flag."""
        eligible = [
            f for f in result.faces
            if f.width >= self._min_face_size and f.height >= self._min_face_size
        ]

        if not eligible:
            return DetectionResult(
                faces=eligible, method=result.method, accepted=False,
                reject_reason="no_face_detected",
            )

        if len(eligible) > self._max_faces:
            return DetectionResult(
                faces=eligible, method=result.method, accepted=False,
                reject_reason=f"too_many_faces:{len(eligible)}",
            )

        return DetectionResult(faces=eligible, method=result.method, accepted=True)

    def largest_face(self, result: DetectionResult) -> FaceBox | None:
        """Return the largest detected face by bounding box area."""
        if not result.faces:
            return None
        return max(result.faces, key=lambda f: f.area)


# ── Cached model loaders ──────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_mtcnn():
    from facenet_pytorch import MTCNN
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MTCNN(keep_all=True, device=device)


@lru_cache(maxsize=1)
def _load_insightface():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(allowed_modules=["detection"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app
