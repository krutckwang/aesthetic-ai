"""Tests for crawler/validation/face_similarity.py — Layer 3 ArcFace validator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from crawler.validation.face_similarity import (
    FaceSimilarityResult,
    FaceSimilarityValidator,
    _cosine,
)


VALIDATOR = FaceSimilarityValidator(min_cosine=0.40)


class TestCosine:
    def test_identical_embeddings_score_one(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert abs(_cosine(v, v) - 1.0) < 1e-5

    def test_opposite_embeddings_score_minus_one(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        assert abs(_cosine(v, -v) - (-1.0)) < 1e-5

    def test_zero_vector_returns_zero(self):
        z = np.zeros(3, dtype=np.float32)
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert _cosine(z, v) == 0.0


class TestBypassWhenInsightFaceUnavailable:
    def test_bypass_mode_when_insightface_missing(self, tmp_path):
        before = tmp_path / "before.jpg"
        after = tmp_path / "after.jpg"
        before.write_bytes(b"fake")
        after.write_bytes(b"fake")

        with patch(
            "crawler.validation.face_similarity._load_app", return_value=None
        ):
            result = VALIDATOR.validate(before, after)

        assert result.cosine_similarity == -2.0
        assert result.same_person is True  # bypass — not a rejection


class TestFaceDetectionFailure:
    def _make_mock_app(self, before_faces, after_faces):
        app = MagicMock()
        app.get.side_effect = [before_faces, after_faces]
        return app

    def test_no_face_in_before_returns_false(self, tmp_path):
        before = tmp_path / "before.jpg"
        after = tmp_path / "after.jpg"
        before.write_bytes(b"fake")
        after.write_bytes(b"fake")

        import cv2
        import numpy as np

        with patch(
            "crawler.validation.face_similarity._load_app",
            return_value=self._make_mock_app([], [MagicMock()]),
        ), patch("cv2.imread", return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
            result = VALIDATOR.validate(before, after)

        assert result.before_face_detected is False
        assert result.same_person is False

    def test_no_face_in_after_returns_false(self, tmp_path):
        before = tmp_path / "before.jpg"
        after = tmp_path / "after.jpg"
        before.write_bytes(b"fake")
        after.write_bytes(b"fake")

        import numpy as np

        mock_face = MagicMock()
        mock_face.bbox = np.array([10.0, 10.0, 200.0, 200.0])
        mock_face.embedding = np.ones(512, dtype=np.float32)
        mock_face.det_score = 0.95

        with patch(
            "crawler.validation.face_similarity._load_app",
            return_value=self._make_mock_app([mock_face], []),
        ), patch("cv2.imread", return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
            result = VALIDATOR.validate(before, after)

        assert result.after_face_detected is False
        assert result.same_person is False


class TestSamePerson:
    def _mock_pair(self, sim: float, tmp_path: Path):
        """Set up a mock InsightFace app that returns embeddings with given cosine sim."""
        import numpy as np

        emb_a = np.random.rand(512).astype(np.float32)
        emb_a /= np.linalg.norm(emb_a)

        # Build emb_b at angle such that cos(emb_a, emb_b) ≈ sim
        perp = np.random.rand(512).astype(np.float32)
        perp -= perp.dot(emb_a) * emb_a
        perp /= np.linalg.norm(perp)

        cos_theta = np.clip(sim, -1.0, 1.0)
        sin_theta = np.sqrt(1 - cos_theta**2)
        emb_b = cos_theta * emb_a + sin_theta * perp
        emb_b = emb_b.astype(np.float32)

        def make_face(emb):
            face = MagicMock()
            face.bbox = np.array([10.0, 10.0, 200.0, 200.0])
            face.embedding = emb
            face.det_score = 0.95
            return face

        before = tmp_path / "before.jpg"
        after = tmp_path / "after.jpg"
        before.write_bytes(b"fake")
        after.write_bytes(b"fake")

        app = MagicMock()
        app.get.side_effect = [[make_face(emb_a)], [make_face(emb_b)]]
        return before, after, app

    def test_high_similarity_same_person(self, tmp_path):
        before, after, app = self._mock_pair(sim=0.85, tmp_path=tmp_path)
        import numpy as np
        with patch(
            "crawler.validation.face_similarity._load_app", return_value=app
        ), patch("cv2.imread", return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
            result = VALIDATOR.validate(before, after)
        assert result.same_person is True
        assert result.cosine_similarity >= 0.40

    def test_low_similarity_different_person(self, tmp_path):
        before, after, app = self._mock_pair(sim=0.10, tmp_path=tmp_path)
        import numpy as np
        with patch(
            "crawler.validation.face_similarity._load_app", return_value=app
        ), patch("cv2.imread", return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
            result = VALIDATOR.validate(before, after)
        assert result.same_person is False
        assert result.cosine_similarity < 0.40

    def test_result_within_valid_range(self, tmp_path):
        before, after, app = self._mock_pair(sim=0.60, tmp_path=tmp_path)
        import numpy as np
        with patch(
            "crawler.validation.face_similarity._load_app", return_value=app
        ), patch("cv2.imread", return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
            result = VALIDATOR.validate(before, after)
        assert -1.0 <= result.cosine_similarity <= 1.0


class TestMissingFile:
    def test_missing_before_file_returns_false(self, tmp_path):
        after = tmp_path / "after.jpg"
        after.write_bytes(b"fake")
        mock_app = MagicMock()
        mock_app.get.return_value = []

        with patch(
            "crawler.validation.face_similarity._load_app", return_value=mock_app
        ):
            result = VALIDATOR.validate(tmp_path / "nonexistent.jpg", after)

        assert result.before_face_detected is False
        assert result.same_person is False
