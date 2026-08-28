"""Tests for crawler/validation/nlp_pairing.py — Layer 2 NLP pairing validator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from crawler.validation.nlp_pairing import (
    NLPPairingResult,
    NLPPairingValidator,
    _detect_language,
    _cosine,
)


VALIDATOR = NLPPairingValidator()


class TestLanguageDetection:
    def test_korean_text_detected(self):
        assert _detect_language("시술 전 사진입니다") == "ko"

    def test_portuguese_text_detected(self):
        assert _detect_language("antes do tratamento estético") == "pt"

    def test_english_default(self):
        assert _detect_language("before and after photos") == "en"

    def test_mixed_korean_english(self):
        # Korean characters should dominate
        assert _detect_language("Before 시술 전") == "ko"


class TestCosine:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.5, 0.3])
        assert abs(_cosine(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine(a, b)) < 1e-6

    def test_zero_vector_returns_zero(self):
        assert _cosine(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0


class TestKeywordFastPath:
    def test_clear_before_after_english_keywords(self):
        result = VALIDATOR.validate(
            before_text="Before botox treatment photos",
            after_text="After treatment results",
        )
        assert result.pairing_score >= 0.85
        assert result.method == "keyword"

    def test_korean_keywords_trigger_fast_path(self):
        result = VALIDATOR.validate(
            before_text="시술전 사진",
            after_text="시술후 결과",
        )
        assert result.pairing_score >= 0.85
        assert result.method == "keyword"

    def test_portuguese_keywords_trigger_fast_path(self):
        result = VALIDATOR.validate(
            before_text="antes do procedimento",
            after_text="resultado após tratamento",
        )
        assert result.pairing_score >= 0.85
        assert result.method == "keyword"

    def test_partial_match_does_not_trigger_fast_path(self):
        """Only one side matching keywords should NOT give keyword score."""
        result = VALIDATOR.validate(
            before_text="before the procedure",
            after_text="random unrelated caption text",
        )
        # Should fall through to semantic or return low score
        assert result.method != "keyword"


class TestEmptyText:
    def test_both_empty_gives_zero_score(self):
        result = VALIDATOR.validate("", "")
        assert result.pairing_score == 0.0
        assert result.method == "none"

    def test_one_empty_goes_to_semantic(self):
        # With one empty and one populated, should attempt semantic
        result = VALIDATOR.validate("before treatment", "")
        assert result.pairing_score >= 0.0  # no crash


class TestSemanticPath:
    def test_semantic_path_invoked_without_keywords(self):
        """Text with no keywords should use semantic path if model available."""
        with patch(
            "crawler.validation.nlp_pairing._load_model"
        ) as mock_load:
            mock_model = MagicMock()
            # Return plausible embeddings: before≈before_template, after≈after_template
            emb = np.random.rand(4, 384).astype(np.float32)
            # Make before text similar to before template, after to after template
            emb[0] = emb[2] + np.random.rand(384) * 0.1   # before_text ≈ before_template
            emb[1] = emb[3] + np.random.rand(384) * 0.1   # after_text ≈ after_template
            mock_model.encode.return_value = emb
            mock_load.return_value = mock_model

            result = VALIDATOR.validate(
                before_text="The patient presented with moderate forehead lines.",
                after_text="Significant improvement observed three weeks post-injection.",
            )
            assert result.method == "semantic"
            assert result.pairing_score >= 0.0

    def test_model_unavailable_gives_partial_score(self):
        """If model is None, partial keyword fallback applies."""
        with patch("crawler.validation.nlp_pairing._load_model", return_value=None):
            result = VALIDATOR.validate(
                before_text="before treatment",
                after_text="some other text without keywords",
            )
            assert result.pairing_score >= 0.0  # no crash
            assert result.method == "keyword_partial"

    def test_encoding_exception_returns_zero(self):
        """An encoding error should return score=0 without raising."""
        with patch("crawler.validation.nlp_pairing._load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.encode.side_effect = RuntimeError("ONNX error")
            mock_load.return_value = mock_model

            result = VALIDATOR.validate(
                before_text="no keywords here just text",
                after_text="more random text",
            )
            assert result.pairing_score == 0.0
            assert result.method == "error"


class TestExtractImageContext:
    def _make_html(self, url: str, alt: str, caption: str) -> str:
        return f"""
        <html><body>
          <figure>
            <img src="{url}" alt="{alt}" />
            <figcaption>{caption}</figcaption>
          </figure>
        </body></html>
        """

    def test_extracts_alt_text(self):
        url = "https://example.com/img.jpg"
        html = self._make_html(url, "before treatment photo", "")
        context = NLPPairingValidator.extract_image_context(url, html)
        assert "before treatment photo" in context

    def test_extracts_figcaption(self):
        url = "https://example.com/img.jpg"
        html = self._make_html(url, "", "Result after botox injection")
        context = NLPPairingValidator.extract_image_context(url, html)
        assert "Result after botox injection" in context

    def test_returns_empty_for_unknown_url(self):
        context = NLPPairingValidator.extract_image_context(
            "https://other.com/missing.jpg",
            "<html><body><img src='https://example.com/img.jpg'/></body></html>",
        )
        assert context == ""

    def test_context_truncated_to_300_chars(self):
        url = "https://example.com/img.jpg"
        long_alt = "before " * 100
        html = f'<html><body><img src="{url}" alt="{long_alt}" /></body></html>'
        context = NLPPairingValidator.extract_image_context(url, html)
        assert len(context) <= 300
