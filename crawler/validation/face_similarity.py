"""
Layer 3 — ArcFace face identity similarity validator.

Uses InsightFace buffalo_l (ArcFace backbone) to verify that the two faces in
a before/after pair belong to the same person.  A low cosine similarity means
the images are of different people (or no detectable face).

buffalo_l is loaded once and cached for the process lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


# ── Thresholds ────────────────────────────────────────────────────────────────
# Conservative uncalibrated default. Will be tuned in calibration.py.
DEFAULT_MIN_COSINE = 0.40


@dataclass
class FaceSimilarityResult:
    """Result of Layer 3 ArcFace face identity check."""

    cosine_similarity: float          # −1.0 to 1.0 (−2.0 = no face detected)
    same_person: bool                 # cosine_similarity >= threshold
    before_face_detected: bool
    after_face_detected: bool
    before_detection_score: float     # detector confidence (0.0 if no face)
    after_detection_score: float


@lru_cache(maxsize=1)
def _load_app(model_dir: str = "") -> "insightface.app.FaceAnalysis | None":
    """Load InsightFace FaceAnalysis app once and cache it."""
    try:
        import insightface
        from insightface.app import FaceAnalysis

        root = model_dir or str(Path.home() / ".insightface")
        app = FaceAnalysis(
            name="buffalo_l",
            root=root,
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("[face_similarity] InsightFace buffalo_l loaded.")
        return app
    except ImportError:
        logger.warning("[face_similarity] insightface not installed — Layer 3 disabled.")
        return None
    except Exception as exc:
        logger.warning(f"[face_similarity] Failed to load buffalo_l: {exc}")
        return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _load_image(path: Path) -> np.ndarray | None:
    """Load image from disk as BGR uint8 array."""
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    if img is None:
        logger.debug(f"[face_similarity] cv2 failed to load {path}")
    return img


class FaceSimilarityValidator:
    """
    Computes ArcFace cosine similarity between the primary face in two images.

    If InsightFace is unavailable, returns a result with cosine_similarity=-2.0
    so the validation pipeline can treat it as a bypass (not a failure).
    """

    def __init__(
        self,
        min_cosine: float = DEFAULT_MIN_COSINE,
        model_dir: str = "",
    ) -> None:
        self.min_cosine = min_cosine
        self._model_dir = model_dir

    def validate(
        self,
        before_path: Path,
        after_path: Path,
    ) -> FaceSimilarityResult:
        """
        Compute ArcFace cosine similarity between the largest face in each image.

        Args:
            before_path: Path to the 'before' image file.
            after_path:  Path to the 'after' image file.

        Returns:
            FaceSimilarityResult with cosine_similarity in [−1, 1], or −2 on error.
        """
        app = _load_app(self._model_dir)

        before_img = _load_image(before_path)
        after_img = _load_image(after_path)

        if app is None:
            # InsightFace unavailable — bypass (not a rejection)
            return FaceSimilarityResult(
                cosine_similarity=-2.0,
                same_person=True,  # bypass mode: don't reject
                before_face_detected=False,
                after_face_detected=False,
                before_detection_score=0.0,
                after_detection_score=0.0,
            )

        before_emb, before_det_score = self._get_face_embedding(app, before_img)
        after_emb, after_det_score = self._get_face_embedding(app, after_img)

        before_detected = before_emb is not None
        after_detected = after_emb is not None

        if not before_detected or not after_detected:
            return FaceSimilarityResult(
                cosine_similarity=0.0,
                same_person=False,
                before_face_detected=before_detected,
                after_face_detected=after_detected,
                before_detection_score=before_det_score,
                after_detection_score=after_det_score,
            )

        sim = _cosine(before_emb, after_emb)
        return FaceSimilarityResult(
            cosine_similarity=round(sim, 4),
            same_person=sim >= self.min_cosine,
            before_face_detected=True,
            after_face_detected=True,
            before_detection_score=before_det_score,
            after_detection_score=after_det_score,
        )

    @staticmethod
    def _get_face_embedding(
        app: "insightface.app.FaceAnalysis",
        img: np.ndarray | None,
    ) -> tuple[np.ndarray | None, float]:
        """
        Detect faces and return the ArcFace embedding of the largest detected face.
        Returns (embedding, detection_score) or (None, 0.0) if no face found.
        """
        if img is None:
            return None, 0.0
        try:
            faces = app.get(img)
        except Exception as exc:
            logger.debug(f"[face_similarity] Detection error: {exc}")
            return None, 0.0

        if not faces:
            return None, 0.0

        # Pick the face with the largest bounding box area
        best = max(
            faces,
            key=lambda f: (
                (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                if hasattr(f, "bbox") and f.bbox is not None
                else 0
            ),
        )

        emb = getattr(best, "embedding", None)
        det_score = float(getattr(best, "det_score", 0.0))

        if emb is None:
            return None, det_score

        return emb.astype(np.float32), det_score
